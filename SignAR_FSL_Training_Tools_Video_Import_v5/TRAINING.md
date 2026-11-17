# SignAR FSL training guide

This guide covers collecting landmark samples, training a TensorFlow model, checking its candidate status, and deciding whether it is safe to deploy.

## Train directly in the browser

The collector automatically imports a selected clip after its required label, verification, and consent details are complete. Very short clips are accepted when at least two usable landmark frames can be extracted; both the source video and landmarks are saved locally. Camera recordings are also collected locally. Saving a sample does not start training. Once at least one labeled sample exists, select **Start Training** to run TensorFlow.js and see epoch, accuracy, and loss updates. A one-label model can only score that sign and cannot distinguish other signs. The model and label metadata are saved to this browser's IndexedDB; there is no manual model or dataset ZIP download.

The browser trainer reports epoch accuracy from the model-fit logs, avoiding extra full-dataset prediction passes after each epoch.

Browser training produces a local candidate marked `trained: false`: its validation is sample-level, not signer-independent, and a one-label model cannot distinguish signs. It is not deployed to the public decoder and cannot be made deployment-ready from an arbitrary upload alone. Verify labels and consent, then use the command-line trainer and signer-independent evaluation before considering deployment. Exported landmark JSON is an optional input to that workflow; source videos remain in browser storage unless downloaded individually.

To try the model, start the camera after training and select **Try a sign**. Perform one fresh sign for about two seconds; the collector runs the saved model on those camera frames and shows its best match without adding the test to training data. Repeat with several performances and signers not used for training. This live check is a sanity check, not a reliable accuracy measurement: the model always returns a best match, and its score is not calibrated confidence.

The public `SignLanguageAR` decoder does not load browser-trained candidates and has no test/try mode. Browser-trained candidates remain local to the collection tool and are not deployable. Only a reviewed, signer-independent model exported in the public decoder's required format should be installed for a release; the decoder rejects manifests with fewer than two labels.

To confirm a run completed, check for **40 / 40 epochs**, the `Training complete · model verified` progress label, and the success message that says the saved model reloaded. These show the fit finished and the stored model artifact is readable. Accuracy and loss are useful diagnostics, but they do not prove recognition quality or deployment readiness; for meaningful evaluation, reserve unseen signers and review labels/results with an FSL expert before testing in the target application.

## Quick start on Windows

Export a dataset to `fsl-dataset.json` in this toolkit folder. Participant datasets are not included in the repository. From the repository root, run:

```powershell
Set-Location .\SignAR_FSL_Training_Tools_Video_Import_v5
.\train.bat -Dataset .\fsl-dataset.json
```

The first run creates `.venv` and installs the pinned dependencies; later runs can skip that step:

```powershell
.\train.bat -Dataset .\fsl-dataset.json -SkipInstall
```

You can also double-click `train.bat` when the dataset is named `fsl-dataset.json` in this folder. The launcher writes a readiness report and starts training even if labels, signer coverage, verification, or consent metadata are incomplete. It writes the model and reports to `trained-model\`.

To pass a dataset exported somewhere else, use its full path:

```powershell
.\train.bat -Dataset "C:\path\to\dataset.json"
```

Training requires at least one labeled sample with a valid feature sequence: at least two finite frames, each containing exactly 456 values. Invalid samples are skipped. The run stops if there are no usable labeled samples or if a required software dependency is missing.

## What the results mean

Training completing means the model fit and export ran; it does **not** mean the model recognizes FSL. A one-label dataset can train a one-output model, but cannot teach the model to distinguish signs. The local five-sample dataset, when present, contains only `KAMUSTA KA`; the pipeline can run, but its accuracy is not a meaningful sign-recognition score and its model is not deployable.

Open the reports after a run:

```powershell
Get-Content .\trained-model\evaluation.json | ConvertFrom-Json |
  Select-Object testAccuracy, minimumClassRecall, evaluationMeaningful, signerIndependent, deploymentEligible

Get-Content .\trained-model\vocabulary.json | ConvertFrom-Json |
  Select-Object trained, testMode, outputClasses, labels
```

- `trained: true` means the candidate passed the configured deployment checks.
- `trained: false` means a model may still have been trained and exported, but it did not pass those checks. Do not copy it to the public decoder.
- `evaluationMeaningful: false` means the evaluation cannot support a claim about recognizing or generalizing to new signers (for example, there is only one label or the split used the same signers).
- `deploymentEligible: false` means required data/evaluation evidence is missing, even if a test metric looks high.

The readiness report (`trained-model\dataset-report.json`) is advisory during a normal training run. `readyForSignerIndependentTraining: false` does not stop training; it explains why the result may not be suitable for deployment. Run validation manually when you want to inspect it:

```powershell
.\.venv\Scripts\python.exe training\validate_dataset.py --dataset .\fsl-dataset.json
```

Add `--strict` only when you specifically want the validator to return a failing exit code for a dataset that is not ready for signer-independent evaluation. It is not needed to train.

## Collecting data

1. Start the local collector from the toolkit directory:
   ```powershell
   python -m http.server 8001
   ```
2. Open `http://localhost:8001/training/collect.html`.
3. Follow the collector's guide: prepare a signer alias and verified sign label, confirm the reference and informed consent, then start the camera or expand **Already have a video?** to choose one sign by one signer.
4. A selected clip imports automatically once the required details are complete. Camera recordings are saved when you stop; neither action starts training.
5. After saving at least one labeled sample, select **Start Training** and monitor the live metrics. To recognize multiple signs, collect a separate label for each sign. The model candidate is stored in this browser. Export landmark JSON only if you want to use the command-line trainer; its label list is derived from the samples included in that export, not a previous model vocabulary. JSON export does not include raw videos.

The collector stores recordings in browser-local storage and does not upload them. Browser storage can be cleared or evicted, so download individual source videos if you need separate copies. Use recordings only with permission, and prefer labels reviewed with a qualified FSL signer, teacher, or interpreter. Training does not validate consent or confirm that a label matches the sign. Keep raw recordings and datasets private.

For meaningful multi-class learning, collect more than one label and multiple varied examples of each label. For a useful unseen-signer evaluation, each label needs examples from multiple signers. Recording each label across at least five signers is a practical target for the current 60/20/20 signer split. These are quality and evaluation recommendations, not prerequisites for starting training.

## Manual setup and training

Use Python 3.10 or newer. From the toolkit folder, the Windows launcher above is recommended. To install or run the commands manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps tensorflowjs==4.22.0
.\.venv\Scripts\python.exe training\train_model.py --dataset .\fsl-dataset.json --output-dir .\trained-model
```

The separate TensorFlow.js converter install is needed on Windows. The toolkit pins compatible TensorFlow/Keras versions; avoid installing the converter with its dependencies on Windows.

Useful trainer options:

| Option | Default | Description |
|---|---:|---|
| `--dataset` | `fsl-dataset.json` | Dataset JSON to train from |
| `--output-dir` | `trained-model` | Model and report output folder |
| `--epochs` | `80` | Maximum epochs; early stopping may finish sooner |
| `--batch-size` | `32` | Samples per training batch |
| `--seed` | `42` | Reproducible sample split and augmentation |
| `--min-test-accuracy` | `0.75` | Accuracy threshold for deployment eligibility |
| `--min-class-recall` | `0.40` | Minimum per-class recall for deployment eligibility |
| `--test-mode` | off | Marks the exported files as pipeline-test artifacts; never deploy them |

Example short run:

```powershell
.\train.bat -Dataset .\fsl-dataset.json -Epochs 1 -SkipInstall
```

For a one-off test using synthetic data, create the fixture and run the test mode:

```powershell
.\.venv\Scripts\python.exe training\create_synthetic_dataset.py
.\train.bat -Dataset .\training\test_synthetic_dataset.json -TestMode -Epochs 1 -SkipInstall
```

Test-mode artifacts go to `test-model\` by default and always have `trained: false`. Synthetic training is only a software-pipeline check.

## How training and evaluation work

The trainer uses every labeled sample with a valid feature sequence; verification and consent metadata do not filter samples out. It resamples variable-duration sequences to 30 frames × 456 features, preserving the `-3` missing-landmark sentinel, and applies mild sign-preserving augmentation to training data only.

It prefers a signer-independent split of about 60% train, 20% validation, and 20% test. If the dataset cannot support that, it falls back to a sample-level split, continues training, and marks the evaluation as non-independent. Training output includes:

- `model.json` and TensorFlow.js weight shards.
- `vocabulary.json` with the labels and `trained` status.
- `evaluation.json` with accuracy, class recall, confusion matrix, and deployment eligibility.
- `dataset-report.json` with validation/readiness findings.

The model uses a softmax output for multiple labels. For a single label, it uses a sigmoid output; this can exercise training but cannot discriminate among signs.

## Deployment

The trainer sets `trained: true` only when the data has multiple labels, readiness checks pass, the evaluation is signer-independent, and configured accuracy and per-class recall thresholds pass. A pipeline test is never deployable. The toolkit accepts arbitrary dataset vocabularies; the public decoder requires at least two labels and a compatible manifest/output order. Before installing a candidate, verify its label list and class order against the decoder. Even a compatible candidate that passes numeric checks is not proof of real-world FSL accuracy: review the signs and per-class results with FSL experts and test with new signers and devices.

Only after that review, copy `trained-model\model.json`, its weight shards, and `trained-model\vocabulary.json` into `SignLanguageAR\model\`. Keep `modelPath` set to `./model/model.json`. Do not copy raw videos or participant datasets into the public repository.

## Troubleshooting

- **Dataset check warnings:** These do not block normal training. Review `dataset-report.json`; the warnings explain evaluation/deployment limitations.
- **No usable samples:** Each included sample needs a non-empty label and at least two finite 456-value feature frames.
- **Missing Python packages:** Use `.\train.bat` to set up dependencies, or install them with the commands above. Add `-SkipInstall` only when the environment is already complete.
- **Browser training engine will not load:** The collector needs an internet connection for its pinned TensorFlow.js CDN script. Check the browser console/network panel, then refresh the page.
- **Training finishes but `trained` is false:** The model was exported, but the data/evaluation did not meet deployment criteria. Use more labels, varied examples, and additional signers; then retrain.
- **TensorFlow.js export fails:** Check the reported TensorFlow and tensorflow.js versions. On Windows, install `tensorflowjs` with `--no-deps` as shown above.

## Related files

- `training/train_model.py` — trainer and TensorFlow.js exporter.
- `training/browser-trainer.js` — on-device TensorFlow.js trainer used by the collection page.
- `training/test_browser_trainer.js` — browser trainer data preparation and sample split tests.
- `training/validate_dataset.py` — optional dataset/readiness report.
- `training/augmentation.py` — training-only augmentation.
- `train.bat` / `train.ps1` — Windows training launcher.
- `requirements.txt` — pinned training dependencies.
