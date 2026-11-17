# SignAR

SignAR is a Filipino Sign Language (FSL) learning and recognition project. This workspace contains two related applications with different purposes and data boundaries.

## Project map

| Area | Purpose | Start here |
|---|---|---|
| [SignLanguageAR/](./SignLanguageAR/) | Public, static browser decoder and learning UI | [Decoder guide](./SignLanguageAR/README.md) |
| [SignAR_FSL_Training_Tools_Video_Import_v5/](./SignAR_FSL_Training_Tools_Video_Import_v5/) | Private, local-first sample collection, validation, and model training; not part of the public deployment | [Toolkit guide](./SignAR_FSL_Training_Tools_Video_Import_v5/README.md) |
| Training workflow | Prepare and evaluate a model before a reviewed release | [Training guide](./SignAR_FSL_Training_Tools_Video_Import_v5/TRAINING.md) |

### Decoder layout

```text
SignLanguageAR/
  index.html, style.css, script.js   Browser application
  shared/                            Browser feature extraction and sequence sampling
  model/vocabulary.json              Public model contract; no trained model by default
  icons/, manifest.webmanifest       Installable web app assets
  sw.js                              Service worker
```

### Training toolkit layout

```text
SignAR_FSL_Training_Tools_Video_Import_v5/
  training/collect.html              Local data collector
  training/train_model.py            Model training and TensorFlow.js export
  training/validate_dataset.py       Dataset checks and readiness report
  training/augmentation.py           Training-only data augmentation
  shared/fsl-features.js             Collector feature extractor
  train.bat, train.ps1               Windows setup, validation, and training launchers
  TRAINING.md                        Detailed training and deployment workflow
```

This repository includes the decoder, local training-tool source, tests, and project documentation for school review. Deploy only `SignLanguageAR/` as the public website; do not expose the training collector on the deployed site. The top-level `.gitignore` excludes local environments, generated model artifacts, sample datasets, reports, and videos. Source tests and synthetic-data generators remain available in GitHub for review.

### Deploy to GitHub Pages

The [Pages workflow](./.github/workflows/pages.yml) publishes only `SignLanguageAR/`; the toolkit source and tests remain in the repository but are not served as the website. In the GitHub repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**. Push to `main` or run the workflow manually to publish the decoder.

## Quick start

### Run the public decoder

Serve [SignLanguageAR/](./SignLanguageAR/) from localhost or HTTPS and open the site in a modern browser. The public decoder has no simulation, sample-collection, training, or candidate-try mode. Without a reviewed model installed it provides landmark tracking only; query parameters do not enable predictions. The private-tool source and its tests are present in this repository for review but are not part of the deployed site. See the [decoder guide](./SignLanguageAR/README.md).

### Collect data and train

Run the toolkit's local collector from the workspace root:

```powershell
Set-Location SignAR_FSL_Training_Tools_Video_Import_v5
python -m http.server 8001
```

Open `http://localhost:8001/training/collect.html`. After exporting a dataset, the current Windows training command is:

```powershell
Set-Location .\SignAR_FSL_Training_Tools_Video_Import_v5
.\train.bat -Dataset .\fsl-dataset.json
```

The first run sets up dependencies; later runs can add `-SkipInstall`. The launcher reports readiness and trains even when the data is incomplete. The local five-sample dataset, when present, contains only `KAMUSTA KA`, so it can exercise training but cannot teach the model to distinguish signs; local participant datasets and generated candidates are excluded from GitHub. Use `-TestMode -Epochs 1` only to mark an output as a pipeline-test artifact. See the [training guide](./SignAR_FSL_Training_Tools_Video_Import_v5/TRAINING.md) for collection, evaluation, and review details.

## Shared model contract

The browser decoder and training toolkit use the `signar-fsl-456-v2` schema:

- 456 features per frame: 132 pose, 252 hand, and 72 selected face values.
- 30-frame temporal inputs sampled at 15 FPS.
- Missing hand/face values use `-3`; missing pose visibility uses `0`.
- Vocabulary order comes from the exported dataset and model manifest.

The public decoder does not include a validated trained model by default. It requires a compatible model manifest with at least two labels in the model's output order. The toolkit can train any dataset vocabulary, but only a reviewed model with verified sign forms, consented data, and unseen-signer evaluation should be installed in the decoder. A pipeline test is not evidence that the model recognizes FSL.

## Data and privacy

Raw recordings and exported participant datasets are private and are not included in this repository. Keep them in a private local location, use signer aliases rather than identifying information, and do not copy source videos or raw datasets into GitHub. The collector stores recordings locally in the browser until a user exports them; it does not upload data.

Dataset reports are snapshots from a particular validation run, not the live source of truth. Re-run validation against the exact dataset you intend to train.

## UI design consistency

Every UI update should prioritize clarity and consistency. Keep the decoder and training collector recognizably part of the same product by reusing their blue palette, light/dark theme behavior, Segoe UI body text, Georgia display headings, and shared corner-radius scale. The applications may keep layouts tailored to their distinct workflows; consistency means matching their visual language and interaction patterns, not forcing identical screens. Prefer clear hierarchy, scannable content, and restrained surfaces over decorative effects. Keep icons only when they clarify an action or status.

The current design system uses a restrained blue palette (including `#286aa5` as the primary blue), coordinated light and dark surfaces, 8px control / 12px surface / 16px camera-stage radii, and matching typography across both apps. The recent UI work applied these styles to the decoder and training collector, improved the collection-page layout, added matching theme controls and dark/light modes, and refreshed their icons and favicons. When changing either app, update and verify both sides where the shared design system is affected. The main styles live in [SignLanguageAR/style.css](./SignLanguageAR/style.css) and [SignAR_FSL_Training_Tools_Video_Import_v5/training/collect-theme.css](./SignAR_FSL_Training_Tools_Video_Import_v5/training/collect-theme.css).
