#!/usr/bin/env python3
"""Train an FSL temporal classifier from SignAR Data Studio's exported JSON.

Raw videos are for human review. This trainer consumes landmark sequences only.
It prefers signer-independent splits and falls back to sample-level splits when needed.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from augmentation import augment_training_data
except ImportError as error:
    print(
        "Training dependencies are missing from this Python interpreter. "
        "From the toolkit folder, run "
        r"'.\.venv\Scripts\python.exe -m pip install -r requirements.txt', "
        "then run the trainer with "
        r"'.\.venv\Scripts\python.exe training\train_model.py'.",
        file=sys.stderr,
    )
    raise SystemExit(3) from error

from validate_dataset import _truthy, find_signer_split, inspect_dataset, normalize_label

SEQ_LEN = 30
FEATURES = 456
FEATURE_VERSION = "signar-fsl-456-v2"


def resample_sequence(frames: list, target: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a variable-length signing sequence to target frames."""
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != FEATURES:
        raise ValueError(f"Expected [time, {FEATURES}] features, got {data.shape}")
    if len(data) < 2:
        raise ValueError("Each sample needs at least 2 feature frames")
    if not np.isfinite(data).all():
        raise ValueError("Feature sequence contains NaN or infinite values")
    positions = np.linspace(0, len(data) - 1, target, dtype=np.float32)
    lo = np.floor(positions).astype(np.int32)
    hi = np.minimum(lo + 1, len(data) - 1)
    weight = (positions - lo)[:, None]
    a, b = data[lo], data[hi]
    interpolated = a * (1.0 - weight) + b * weight
    # -3 is the dedicated missing-landmark sentinel; do not blend it into fake coordinates.
    missing = (a <= -2.75) | (b <= -2.75)
    nearest = np.where(weight < 0.5, a, b)
    return np.where(missing, nearest, interpolated)


def load_dataset(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset root must be a JSON object")
    labels = payload.get("labels")
    labels = [normalize_label(label) for label in labels if isinstance(label, str)] if isinstance(labels, list) else []
    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list):
        raise ValueError("Dataset samples must be a JSON array")
    observed_labels = [normalize_label(sample.get("label", "")) for sample in samples_payload
                       if isinstance(sample, dict)]
    labels = list(dict.fromkeys(label for label in labels + observed_labels if label))
    if not labels:
        raise ValueError("No sign labels were found in the dataset or its samples")
    label_set = set(labels)
    samples = []
    invalid = Counter()
    for sample in samples_payload:
        if not isinstance(sample, dict):
            invalid["invalid_sample"] += 1
            continue
        label = normalize_label(sample.get("label", ""))
        signer = str(sample.get("signerId", "")).strip()
        frames = sample.get("features", [])
        if label not in label_set:
            invalid["unknown_label"] += 1
            continue
        try:
            sequence = resample_sequence(frames)
        except (ValueError, TypeError):
            invalid["invalid_features"] += 1
            continue
        samples.append({
            "label": label,
            "signer": signer,
            "x": sequence,
            "id": sample.get("id", ""),
            "verified": _truthy(sample.get("verified", False)),
            "consentRecorded": _truthy(sample.get("consentRecorded", False)),
            "verificationNotes": str(sample.get("verificationNotes", "")).strip(),
        })
    if invalid:
        print("Skipped sample counts:", dict(invalid))
    if not samples:
        raise ValueError("No samples with valid 456-feature sequences and labels found in this dataset JSON")
    return samples, labels


def split_for_test(samples, seed: int):
    """Make a sample-level split when signer-independent evaluation is unavailable."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        return (shuffled, shuffled, shuffled), {"train": [], "validation": [], "test": []}
    train_end = max(1, int(len(shuffled) * 0.6))
    validation_end = max(train_end + 1, int(len(shuffled) * 0.8))
    validation_end = min(validation_end, len(shuffled) - 1)
    parts = (shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:])
    return parts, {"train": [], "validation": [], "test": []}


def split_by_signer(samples, labels: list[str], seed: int):
    """Find a 60/20/20 group split where all classes occur in each partition."""
    signer_labels: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        signer_labels[sample["signer"]].add(sample["label"])
    signers = sorted(signer_labels)
    if len(signers) < 5:
        raise ValueError(f"Need at least 5 distinct signers for signer-independent train/validation/test sets; got {len(signers)}")
    missing_support = [label for label in labels if sum(label in signer_labels[s] for s in signers) < 3]
    if missing_support:
        raise ValueError("Every class must be recorded by at least 3 distinct signers before a fair signer-independent evaluation. Missing coverage: " + ", ".join(missing_support))

    split = find_signer_split(signer_labels, labels, seed)
    if split:
        partitions = [[sample for sample in samples if sample["signer"] in split[name]]
                      for name in ("train", "validation", "test")]
        return partitions, split
    raise ValueError("Could not find a signer-independent split covering all labels. Record every label across more signers, then retry.")


def build_model(num_classes: int):
    import tensorflow as tf

    single_label = num_classes == 1
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, FEATURES), name="landmark_sequence"),
        tf.keras.layers.GRU(96, return_sequences=True, dropout=0.15, name="temporal_gru_1"),
        tf.keras.layers.GRU(64, dropout=0.20, name="temporal_gru_2"),
        tf.keras.layers.Dense(64, activation="relu", name="representation"),
        tf.keras.layers.Dropout(0.25, name="classifier_dropout"),
        tf.keras.layers.Dense(num_classes, activation="sigmoid" if single_label else "softmax",
                              name="fsl_class_probabilities"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy" if single_label else "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("fsl-dataset.json"), help="Exported landmark JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("trained-model"), help="Folder for candidate TensorFlow.js model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-test-accuracy", type=float, default=0.75, help="Accuracy needed to mark candidate manifest trained=true")
    parser.add_argument("--min-class-recall", type=float, default=0.40, help="Minimum recall required for every vocabulary class")
    parser.add_argument("--test-mode", action="store_true", help="Mark output as a pipeline test artifact that must not be deployed")
    args = parser.parse_args()
    try:
        samples, labels = load_dataset(args.dataset)
        try:
            parts, signer_split = split_by_signer(samples, labels, args.seed)
            signer_independent = True
        except ValueError as error:
            parts, signer_split = split_for_test(samples, args.seed)
            signer_independent = False
            print(f"WARNING: {error}")
            print("Using a sample-level split so training can continue; evaluation is not signer-independent.")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DATASET LOAD FAILED: {error}", file=sys.stderr)
        return 2
    if args.test_mode:
        print("TEST MODE: this output is a pipeline test artifact and must not be deployed.")
    readiness = inspect_dataset(json.loads(args.dataset.read_text(encoding="utf-8")))
    deployable_data = (
        readiness["readyForSignerIndependentTraining"]
        and len(labels) >= 2
        and signer_independent
        and all(sample["verified"] and sample["consentRecorded"] and sample["verificationNotes"] for sample in samples)
    )
    evaluation_meaningful = signer_independent and len(labels) >= 2 and not args.test_mode

    try:
        import tensorflow as tf
        if sys.platform == "win32":
            try:
                import tensorflow_decision_forests  # noqa: F401
            except ModuleNotFoundError as error:
                if error.name != "tensorflow_decision_forests":
                    raise
                # TFJS imports TFDF unconditionally, but this GRU model has no TFDF ops.
                sys.modules["tensorflow_decision_forests"] = types.ModuleType(
                    "tensorflow_decision_forests"
                )
        import tensorflowjs as tfjs
    except (ImportError, OSError) as error:
        print(f"TRAINING DEPENDENCY CHECK FAILED: {error}", file=sys.stderr)
        return 3
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    index = {label:i for i,label in enumerate(labels)}
    def arrays(part):
        x = np.stack([s["x"] for s in part]).astype(np.float32)
        y = np.array([index[s["label"]] for s in part], dtype=np.int32)
        return x, y
    (train_x,train_y),(val_x,val_y),(test_x,test_y) = (arrays(p) for p in parts)
    train_x, train_y = augment_training_data(train_x, train_y, args.seed, SEQ_LEN, FEATURES)
    print(f"Samples after mild training-only augmentation: train={len(train_y)}, validation={len(val_y)}, test={len(test_y)}")
    print("Signer split:", json.dumps(signer_split, ensure_ascii=False))
    print("Training label counts:", {labels[i]:int(Counter(train_y)[i]) for i in range(len(labels))})

    counts = np.bincount(train_y, minlength=len(labels)).astype(np.float32)
    class_weights = {i:float(len(train_y)/(len(labels)*max(1,counts[i]))) for i in range(len(labels))}
    model = build_model(len(labels))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.00001),
    ]
    fit_train_y = np.ones_like(train_y) if len(labels) == 1 else train_y
    fit_val_y = np.ones_like(val_y) if len(labels) == 1 else val_y
    model.fit(train_x,fit_train_y,validation_data=(val_x,fit_val_y),epochs=args.epochs,batch_size=args.batch_size,
              class_weight=class_weights,callbacks=callbacks,verbose=2,shuffle=True)
    probabilities = model.predict(test_x,verbose=0)
    predictions = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(test_y,predictions))
    report = classification_report(test_y,predictions,labels=list(range(len(labels))),target_names=labels,
                                   output_dict=True,zero_division=0)
    conf = confusion_matrix(test_y,predictions,labels=list(range(len(labels)))).tolist()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        tfjs.converters.save_keras_model(model,str(args.output_dir))
    except Exception as error:
        print("TensorFlow.js export failed. Check tensorflow/tensorflowjs versions:",error,file=sys.stderr)
        return 3

    minimum_class_recall = min(float(report[label]["recall"]) for label in labels)
    trained_ok = (
        not args.test_mode
        and deployable_data
        and accuracy >= args.min_test_accuracy
        and minimum_class_recall >= args.min_class_recall
    )
    manifest = {
      "project":"SignAR FSL Decoder", "schemaVersion":1, "featureVersion":FEATURE_VERSION,
      "trained":trained_ok, "testMode":args.test_mode, "modelPath":"./model/model.json", "sequenceLength":SEQ_LEN,
      "samplingFps":15, "sampleWindowSeconds":2, "featureDimension":FEATURES,
      "outputClasses":len(labels), "labels":labels,
      "featureSchema":{"pose":{"count":33,"valuesPerPoint":["x","y","z","visibility"],"size":132,"missingPointVisibility":0},
                       "hands":{"countPerHand":21,"valuesPerPoint":["bodyRelativeX","bodyRelativeY","bodyRelativeZ","wristRelativeX","wristRelativeY","wristRelativeZ"],"handCount":2,"size":252,"missingValue":-3},
                       "face":{"pointCount":24,"valuesPerPoint":["x","y","z"],"size":72,"missingValue":-3},"total":FEATURES,
                       "normalization":{"anchorFallbackOrder":["shouldersAndHips","shoulders","selectedFacePoints","partialUpperBody","visibleHands"],
                                        "normalizedCoordinateRange":[-2.5,2.5],"missingValueForHandAndFace":-3,
                                        "missingValueInterpolation":"nearest-sample"}},
      "evaluation":{"signerIndependent":signer_independent,"evaluationMeaningful":evaluation_meaningful,
                     "deploymentEligible":deployable_data and not args.test_mode,
                     "testAccuracy":accuracy,"minimumAccuracy":args.min_test_accuracy,
                     "minimumClassRecall":minimum_class_recall,"requiredClassRecall":args.min_class_recall,
                     "trainSigners":signer_split["train"],"validationSigners":signer_split["validation"],"testSigners":signer_split["test"]},
      "note":("TEST-ONLY model. This artifact must not be deployed."
              if args.test_mode else
              "Review evaluation.json and confirm sign forms with FSL experts before deploying. Incomplete data can train a candidate but does not make it deployable.")
    }
    (args.output_dir/"vocabulary.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    evaluation = {"testMode":args.test_mode,"evaluationMeaningful":evaluation_meaningful,
                  "deploymentEligible":deployable_data and not args.test_mode,
                  "signerIndependent":signer_independent,
                  "testAccuracy":accuracy,"minimumAccuracy":args.min_test_accuracy,
                  "minimumClassRecall":minimum_class_recall,"requiredClassRecall":args.min_class_recall,"passedMinimum":trained_ok,
                  "classificationReport":report,"confusionMatrix":conf,"labels":labels,"signerSplit":signer_split,
                  "sampleCounts":{"train":len(train_y),"validation":len(val_y),"test":len(test_y)}}
    (args.output_dir/"evaluation.json").write_text(json.dumps(evaluation,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    metric_name = "Unseen-signer" if signer_independent else "Sample-level (not signer-independent)"
    print(f"{metric_name} test accuracy: {accuracy:.3f}")
    print(f"Candidate saved to: {args.output_dir.resolve()}")
    if args.test_mode:
        print("TEST-ONLY model exported with trained=false. Do not deploy this artifact.")
        return 0
    if not trained_ok:
        print("Candidate exported with trained=false. Training completed, but dataset coverage/verification or evaluation thresholds are not sufficient for deployment.")
        return 0
    print("Candidate meets the numeric threshold. Review per-class errors and FSL sign validity before copying artifacts into the public model folder.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
