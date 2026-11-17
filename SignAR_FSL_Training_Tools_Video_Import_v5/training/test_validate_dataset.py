"""Lightweight tests for dataset validation; no ML dependencies are needed."""
import json
import tempfile
import unittest
from pathlib import Path

from train_model import load_dataset, split_by_signer, split_for_test
from validate_dataset import FEATURES, FEATURE_VERSION, find_signer_split, inspect_dataset

TEST_LABELS = ["WORD_A", "WORD_B", "WORD_C", "WORD_D", "WORD_E"]


def frame():
    return [0.0] * FEATURES


def complete_payload():
    samples = []
    for signer_index in range(5):
        signer = f"signer-{signer_index+1:02d}"
        for label_index, label in enumerate(TEST_LABELS):
            samples.append({
                "id": f"{signer}-{label_index:02d}",
                "signerId": signer,
                "label": label,
                "verified": True,
                "consentRecorded": True,
                "verificationNotes": "Reference reviewed with FSL teacher",
                "features": [frame(), frame()],
            })
    return {
        "schemaVersion": 1,
        "featureVersion": FEATURE_VERSION,
        "featureDimension": FEATURES,
        "labels": TEST_LABELS,
        "samples": samples,
    }


class ValidateDatasetTests(unittest.TestCase):
    def test_complete_signer_independent_coverage(self):
        report = inspect_dataset(complete_payload())
        self.assertTrue(report["readyForSignerIndependentTraining"])
        self.assertEqual(report["validSamples"], 25)
        self.assertEqual(report["distinctSigners"], 5)
        self.assertEqual(report["featureDimension"], 456)

    def test_unconsented_sample_is_excluded_once(self):
        payload = complete_payload()
        payload["samples"][0]["consentRecorded"] = False
        report = inspect_dataset(payload)
        self.assertEqual(report["excludedSamples"], 1)
        self.assertEqual(report["excludedReasons"]["consent_not_confirmed"], 1)
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_invalid_feature_width_is_excluded(self):
        payload = complete_payload()
        payload["samples"][0]["features"][0] = [0.0] * 455
        report = inspect_dataset(payload)
        self.assertEqual(report["excludedSamples"], 1)
        self.assertEqual(report["excludedReasons"]["invalid_feature_values"], 1)
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_truthy_variants_and_trimmed_labels_are_accepted(self):
        payload = complete_payload()
        for sample in payload["samples"]:
            sample["verified"] = "true"
            sample["consentRecorded"] = 1
            sample["label"] = f" {sample['label']} "
        report = inspect_dataset(payload)
        self.assertTrue(report["readyForSignerIndependentTraining"])
        self.assertEqual(report["validSamples"], 25)

    def test_feature_schema_mismatch_is_error(self):
        payload = complete_payload()
        payload["featureVersion"] = "old-version"
        report = inspect_dataset(payload)
        self.assertTrue(any("featureVersion" in e for e in report["errors"]))
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_split_requires_every_label_in_train_validation_and_test(self):
        payload = complete_payload()
        support = {
            "WORD_A": {"signer-01", "signer-02", "signer-03"},
            "WORD_B": {"signer-03", "signer-04", "signer-05"},
        }
        payload["samples"] = [
            sample for sample in payload["samples"]
            if sample["label"] not in support or sample["signerId"] in support[sample["label"]]
        ]
        report = inspect_dataset(payload)
        self.assertFalse(report["readyForSignerIndependentTraining"])
        self.assertTrue(any("Could not find a signer-independent split" in warning for warning in report["warnings"]))

    def test_shared_split_contains_each_label_in_every_partition(self):
        signer_labels = {f"signer-{i:02d}": set(TEST_LABELS) for i in range(1, 6)}
        split = find_signer_split(signer_labels, TEST_LABELS, seed=7)
        self.assertIsNotNone(split)
        for signer_ids in split.values():
            present = set().union(*(signer_labels[signer] for signer in signer_ids))
            self.assertEqual(present, set(TEST_LABELS))

    def test_non_object_dataset_root_is_reported(self):
        report = inspect_dataset([])
        self.assertTrue(any("root must be a JSON object" in error for error in report["errors"]))
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_trainer_loads_structurally_valid_samples_without_verification_metadata(self):
        payload = complete_payload()
        payload["samples"][0]["verified"] = False
        payload["samples"][0]["consentRecorded"] = False
        payload["labels"] = ["WORD_A"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        self.assertEqual(len(samples), 25)
        self.assertEqual(len(labels), 5)
        self.assertFalse(samples[0]["verified"] and samples[0]["consentRecorded"])

    def test_trainer_loads_one_label_without_signer_aliases(self):
        payload = {
            "labels": ["HELLO"],
            "samples": [{
                "label": "HELLO",
                "features": [frame(), frame()],
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        parts, split = split_for_test(samples, seed=7)
        self.assertEqual(labels, ["HELLO"])
        self.assertEqual(len(samples), 1)
        self.assertEqual([len(part) for part in parts], [1, 1, 1])
        self.assertEqual(split, {"train": [], "validation": [], "test": []})

    def test_test_mode_accepts_unverified_unconsented_single_signer_single_class_data(self):
        payload = {
            "labels": [" sample "],
            "samples": [{
                "label": "sample",
                "features": [frame(), frame()],
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        parts, split = split_for_test(samples, seed=7)
        self.assertEqual(labels, ["SAMPLE"])
        self.assertEqual([len(part) for part in parts], [1, 1, 1])
        self.assertEqual(split, {"train": [], "validation": [], "test": []})

    def test_test_mode_infers_labels_and_still_rejects_bad_feature_shape(self):
        payload = {
            "samples": [
                {"label": "one", "features": [frame(), frame()]},
                {"label": "two", "features": [frame(), frame()]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
            self.assertEqual(labels, ["ONE", "TWO"])
            self.assertEqual(len(samples), 2)

            payload["samples"][0]["features"][0] = [0.0] * 455
            payload["samples"][1]["features"][0] = [0.0] * 455
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No samples with valid 456-feature"):
                load_dataset(path)

    def test_trainer_loads_ready_dataset_and_covers_labels_in_every_partition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(complete_payload()), encoding="utf-8")
            samples, labels = load_dataset(path)
        partitions, _ = split_by_signer(samples, labels, seed=7)
        self.assertEqual(len(samples), 25)
        for partition in partitions:
            self.assertEqual({sample["label"] for sample in partition}, set(TEST_LABELS))

    def test_trainer_refuses_split_missing_a_label_from_one_partition(self):
        signer_labels = {
            "signer-01": {"WORD_A", "WORD_C", "WORD_D", "WORD_E"},
            "signer-02": {"WORD_A", "WORD_C", "WORD_D", "WORD_E"},
            "signer-03": set(TEST_LABELS),
            "signer-04": {"WORD_B", "WORD_C", "WORD_D", "WORD_E"},
            "signer-05": {"WORD_B", "WORD_C", "WORD_D", "WORD_E"},
        }
        samples = [
            {"signer": signer, "label": label}
            for signer, labels in signer_labels.items()
            for label in labels
        ]
        with self.assertRaisesRegex(ValueError, "Could not find a signer-independent split"):
            split_by_signer(samples, TEST_LABELS, seed=7)


if __name__ == "__main__":
    unittest.main()
