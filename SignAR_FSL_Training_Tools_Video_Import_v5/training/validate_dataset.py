#!/usr/bin/env python3
"""Validate a SignAR FSL dataset and report label/signer coverage before training.

Uses only Python's standard library so it can run before TensorFlow is installed.
It never prints landmark arrays or any raw video data.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

FEATURE_VERSION = "signar-fsl-456-v2"
FEATURES = 456


def normalize_label(value: object) -> str:
    """Normalize dataset labels to a consistent uppercase, trimmed form."""
    text = str(value).strip()
    return text.upper() if text else ""


def _truthy(value: object) -> bool:
    """Accept common JSON/CSV truthy encodings without excluding valid samples."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return value == 1
    return False


def find_signer_split(signer_labels: dict[str, set[str]], labels: list[str], seed: int = 42, tries: int = 2000) -> dict | None:
    """Return a valid signer split with every label in every partition, if possible."""
    signers = sorted(signer_labels)
    if len(signers) < 5 or not labels:
        return None
    if any(sum(label in signer_labels[signer] for signer in signers) < 3 for label in labels):
        return None
    n_train = max(3, int(len(signers) * .60))
    n_val = max(1, int(len(signers) * .20))
    if n_train + n_val >= len(signers):
        n_train, n_val = len(signers) - 2, 1

    rng = random.Random(seed)
    order = signers[:]
    rng.shuffle(order)

    for train in combinations(order, n_train):
        remaining = [signer for signer in order if signer not in train]
        if len(remaining) < n_val + 1:
            continue
        for validation in combinations(remaining, n_val):
            validation_set = set(validation)
            test = [signer for signer in remaining if signer not in validation_set]
            if not test:
                continue
            groups = {
                "train": list(train),
                "validation": list(validation),
                "test": test,
            }
            if all(all(any(label in signer_labels[signer] for signer in group) for label in labels)
                   for group in groups.values()):
                return {name: sorted(group) for name, group in groups.items()}
    return None


def signer_split_possible(samples: list[dict], labels: list[str], seed: int = 42, tries: int = 2000) -> bool:
    signer_labels: dict[str, set[str]] = {}
    for sample in samples:
        signer_labels.setdefault(sample["signerId"], set()).add(sample["label"])
    return find_signer_split(signer_labels, labels, seed, tries) is not None


def inspect_dataset(payload: dict) -> dict:
    errors = []
    warnings = []
    if not isinstance(payload, dict):
        payload = {}
        errors.append("dataset root must be a JSON object")
    if payload.get("featureVersion") != FEATURE_VERSION:
        errors.append(f"featureVersion must be {FEATURE_VERSION!r}")
    if payload.get("featureDimension") != FEATURES:
        errors.append(f"featureDimension must be {FEATURES}")
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(l, str) for l in labels):
        errors.append("dataset must declare a non-empty labels array of strings")
        labels = []
    else:
        normalized_labels = []
        seen = set()
        for label in labels:
            clean = normalize_label(label)
            if not clean:
                errors.append("labels must be non-empty strings without surrounding whitespace")
                continue
            if clean in seen:
                errors.append("labels must not contain duplicates after normalization")
                continue
            seen.add(clean)
            normalized_labels.append(clean)
        labels = normalized_labels
        if labels and any(normalize_label(label) != label for label in payload.get("labels", [])):
            warnings.append("Labels were normalized to uppercase without surrounding whitespace.")
        if len(labels) != len(set(labels)):
            errors.append("labels must not contain duplicates")
    label_set = set(labels)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raw_samples = []
        errors.append("samples must be a JSON array")

    ids = set()
    valid = []
    invalid_counts = Counter()
    excluded_sample_count = 0
    for index, sample in enumerate(raw_samples):
        reasons = []
        if not isinstance(sample, dict):
            invalid_counts["sample_not_object"] += 1
            excluded_sample_count += 1
            continue
        sample_id = str(sample.get("id", ""))
        if not sample_id:
            reasons.append("missing_id")
        elif sample_id in ids:
            reasons.append("duplicate_id")
        ids.add(sample_id)
        label = normalize_label(sample.get("label", ""))
        signer = str(sample.get("signerId", "")).strip()
        if label not in label_set:
            reasons.append("unknown_label")
        if not signer:
            reasons.append("missing_signer")
        if not _truthy(sample.get("verified")):
            reasons.append("not_verified")
        if not _truthy(sample.get("consentRecorded")):
            reasons.append("consent_not_confirmed")
        if len(str(sample.get("verificationNotes", "")).strip()) < 5:
            reasons.append("missing_verification_notes")
        features = sample.get("features")
        if not isinstance(features, list) or len(features) < 2:
            reasons.append("too_few_frames")
        else:
            bad_frame = False
            for frame in features:
                if not isinstance(frame, list) or len(frame) != FEATURES:
                    bad_frame = True
                    break
                for value in frame:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        bad_frame = True
                        break
                if bad_frame:
                    break
            if bad_frame:
                reasons.append("invalid_feature_values")
        if reasons:
            excluded_sample_count += 1
            for reason in reasons:
                invalid_counts[reason] += 1
            continue
        valid.append({"id":sample_id,"label":label,"signerId":signer,"frameCount":len(features)})

    label_counts = Counter(s["label"] for s in valid)
    signer_counts = Counter(s["signerId"] for s in valid)
    label_signers = {label: sorted({s["signerId"] for s in valid if s["label"] == label}) for label in labels}
    missing_labels = [label for label in labels if label_counts[label] == 0]
    weak_signer_coverage = {label:len(signers) for label,signers in label_signers.items() if len(signers) < 3}
    single_class = len(labels) == 1
    split_ready = not single_class and not missing_labels and not weak_signer_coverage and signer_split_possible(valid, labels)
    if single_class:
        warnings.append("Single-label data can be trained, but cannot teach the model to distinguish signs.")
    if missing_labels:
        warnings.append("Collect samples for missing labels: " + ", ".join(missing_labels))
    if len(signer_counts) < 5:
        warnings.append(f"Only {len(signer_counts)} signer aliases are represented; evaluation may not generalize to unseen signers.")
    if weak_signer_coverage:
        warnings.append("Some labels have limited signer coverage: " + ", ".join(f"{k} ({v})" for k,v in weak_signer_coverage.items()))
    if not split_ready and not single_class and not missing_labels and not weak_signer_coverage and len(signer_counts) >= 5:
        warnings.append("Could not find a signer-independent split with all labels in every partition. Collect more words from more signers.")
    if invalid_counts:
        warnings.append("Some samples were not counted as evaluation-ready because of metadata or feature checks; the trainer can use structurally valid feature sequences.")

    next_step = (
        "Training can run with the structurally valid samples. Treat readiness and coverage warnings as deployment guidance; review evaluation and sign accuracy before deployment."
    )

    return {
        "schemaVersion":1,
        "featureVersion":payload.get("featureVersion"),
        "featureDimension":payload.get("featureDimension"),
        "samplesInExport":len(raw_samples),
        "validSamples":len(valid),
        "excludedSamples":excluded_sample_count,
        "distinctSigners":len(signer_counts),
        "readyForSignerIndependentTraining":bool(split_ready and not errors and not invalid_counts),
        "labelCounts":{label:label_counts[label] for label in labels},
        "distinctSignersPerLabel":{label:len(label_signers[label]) for label in labels},
        "samplesPerSigner":dict(sorted(signer_counts.items())),
        "excludedReasons":dict(invalid_counts),
        "errors":errors,
        "warnings":warnings,
        "nextStep":next_step
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("fsl-dataset.json"))
    parser.add_argument("--report", type=Path, default=Path("dataset-report.json"), help="Where to save a metadata-only report")
    parser.add_argument("--strict", action="store_true", help="Return nonzero unless the dataset meets deployment-evaluation readiness checks")
    args = parser.parse_args()
    try:
        payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read dataset: {exc}", file=sys.stderr)
        return 2
    report = inspect_dataset(payload)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict:
        if report["errors"]:
            return 2
        if not report["readyForSignerIndependentTraining"]:
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
