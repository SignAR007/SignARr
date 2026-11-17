# SignAR FSL private training toolkit

This toolkit is separate from the public SignAR decoder. Do not upload private recordings or this data tool to the public website. For reliable and responsible collection, confirm that the sign form was checked against a reliable FSL reference and a qualified FSL signer/teacher/interpreter, and that the signer gave informed permission. The trainer can run without this metadata, but such a model is not eligible for deployment.

## 1. Run the local recorder

Requirements: a current browser, Python 3.10+ (for the local HTTP server), and an internet connection for the pinned MediaPipe Holistic and TensorFlow.js CDN files.

1. Extract the ZIP to a local folder.
2. Open a terminal in that folder (the directory containing `training`, `shared`, and `model`).
3. Run `python -m http.server 8001` (this build deliberately uses port 8001 so an older server on port 8000 cannot serve the wrong page).
4. Open `http://localhost:8001/training/collect.html` in your browser. Follow the on-screen guide: prepare the sign details and consent, record or choose clips, then select **Start Training** when your samples are ready.
5. Use an alias such as `signer-01`, choose a label from the saved-sample suggestions or type a new verified sign label, and add a brief reference/checker note. Then confirm verification and consent. The interface enables recording/import when these details are ready and shows what to do next.
6. Record one natural performance. Manual recording is the default; optional **Hands-free capture** starts after detected movement and saves after a short pause.
7. To import a recording instead, expand **Already have a video?**, choose one clip, and finish the sign details and consent if needed. The clip is then analyzed locally at 15 samples/second and saved with its extracted landmarks; no separate import button is needed. Use one sign by one signer per clip; split multi-sign recordings first. The importer accepts a clip with as few as 4 usable landmark frames. MP4 (H.264) and WebM are recommended; support for other formats depends on the browser.
8. Saving or importing a sample does not train. Once at least one labeled sample is collected, select **Start Training**. A one-label model can only score that label; it cannot distinguish other signs. Afterward, start the camera and choose **Try a sign** to check a fresh performance; it shows the model's best match without saving that test. Repeat with performances and signers excluded from training for a more useful check. The model and labels stay local; no dataset ZIP, model ZIP, or terminal commands are needed. Training completion and a best-match score do not establish recognition accuracy or deployment readiness.
9. **Export landmark JSON** is optional for the separate command-line trainer. It contains landmarks and metadata, not the source videos. Source videos remain in this browser unless you download them individually from the sample list.

This v5 build uses a local IndexedDB store for the v2 feature schema so old v1 samples are not mislabeled. Previously saved v1 samples remain in the older toolkit's store; re-import their source videos with v5 to generate compatible features. Samples are stored in this browser's IndexedDB. Clearing browser site data can remove them, and browser storage limits vary. Download source videos individually if you need separate copies; landmark JSON exports do not include videos. If camera permission fails, ensure you opened the tool on `localhost` (not as a `file://` page).

The collector and decoder use matching high-accuracy Holistic settings: highest pose model complexity and input up to 1280 px. If frame processing exceeds 65 ms for three consecutive frames, resolution steps down (1280 → 960 → 720) and recovers after 120 frames under 42 ms. Live camera cadence also adapts from 12 to 30 FPS, balancing detail and responsiveness across devices.

## 2. Collect a signer-independent dataset

Vocabulary is taken from the `labels` array declared in your exported `fsl-dataset.json`, not hardcoded. Set whatever word/sign labels you need in the collector, then re-export the dataset JSON. Each label must be uppercase and match exactly between samples.

These are candidate labels, not an assertion that every sign form has already been validated. Record right-handed signers facing the camera when possible. The importer does not require the full body, both hands, or every selected facial landmark. It uses a torso anchor when shoulders are available, and falls back to selected face points, a partial upper-body pose, or visible hands when needed. Frames with no meaningful pose, hand, or selected face points are still rejected. Vary lighting, distance, clothing, and background across sessions. Use multiple performances per signer/word; avoid recording only one clean performance. Use anonymous aliases rather than names.

Those are data-quality goals, not training prerequisites: training can start with one label, one signer, or incomplete verification/consent metadata. At least two labels are needed to learn to distinguish signs; an unseen-signer evaluation also needs enough signers represented in each split. As a practical starting target, record every label from multiple participating signers, with 5–10 varied samples per label per signer if feasible. More independent signers generally provide a stronger test of generalization.

## 3. Train the temporal model

### Windows quick start

After exporting `dataset.json` from the collector, either copy it here as `fsl-dataset.json` and double-click `train.bat`, or pass the exported file directly:

```powershell
.\train.bat -Dataset "C:\path\to\dataset.json"
```

The launcher creates `.venv` if needed, installs dependencies, writes a dataset-readiness report, and starts training even when coverage or verification warnings are present. It writes the report and candidate to `trained-model\`. Weak or incomplete data can train, but the result stays `trained: false` unless it also passes signer-independent evaluation and the deployment checks. The local five-sample dataset, when present, contains only `KAMUSTA KA`; it can exercise training but cannot distinguish signs and is not deployable. Participant datasets are intentionally omitted from this repository; export a dataset you have permission to use and save it as `fsl-dataset.json` or pass its path. Training does not validate consent or sign correctness. For pipeline testing, use `.\train.bat -Dataset "C:\path\to\dataset.json" -TestMode -Epochs 1`; test output goes to `test-model\` and is never deployable. For an existing environment, add `-SkipInstall`.

### Manual setup

Use Python 3.10 or newer. From this directory:

```bash
py -3.11 -m venv .venv
# Windows PowerShell; use this interpreter explicitly so global Python is not selected.
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Windows only: install the converter without TFDF, which has no Windows wheel.
.\.venv\Scripts\python.exe -m pip install --no-deps tensorflowjs==4.22.0
```

The Windows requirements pin matching TensorFlow/Keras versions to avoid native DLL and protobuf conflicts. TensorFlow.js imports TensorFlow Decision Forests even for models that do not use it; this GRU classifier has no TFDF operations, so the Windows converter is installed without dependencies and the trainer skips only that unused import.

Copy the exported landmark JSON to this folder as `fsl-dataset.json`. Optionally inspect data quality and signer/word coverage with:

```bash
.\.venv\Scripts\python.exe training\validate_dataset.py --dataset fsl-dataset.json
```

Run training; readiness warnings are guidance for evaluation and deployment, not a prerequisite:

```bash
.\.venv\Scripts\python.exe training\train_model.py --dataset fsl-dataset.json --output-dir trained-model
```

The script resamples variable-duration clips to 30 frames, applies only mild scale/rotation/coordinate-noise/timing augmentation to the training set (never to validation or test data), trains a GRU-based temporal classifier, and tests it on unseen signers where the dataset permits. If not, it falls back to a sample-level split. It never mirrors signs or changes their labels. It writes TensorFlow.js files, `evaluation.json`, and a candidate `vocabulary.json` in `trained-model/`. The candidate manifest marks the model `trained=true` only when the data/readiness checks, signer-independent coverage, 75% test accuracy, and 40% minimum per-class recall all pass. Otherwise training still completes and the output remains `trained=false`. The trainer accepts arbitrary dataset vocabularies; the public decoder requires at least two labels and a compatible manifest/output order. Review results and FSL sign forms before deployment.

On Windows, running `python training\train_model.py` may select a separate system Python without the training packages. Use the explicit `.venv\Scripts\python.exe` commands above, or activate `.venv` before running `python`.

## 4. Publish only after review

Do not copy raw videos or participant datasets into the public repository. Once the candidate has been reviewed, copy the generated `model.json`, weight shard files, and `vocabulary.json` into `SignLanguageAR/model/` on a feature branch. Check that the manifest `modelPath` remains `./model/model.json`. Test on actual devices and new signers before merging.

## Data schema

- The collector and decoder keep matching `shared/fsl-features.js` implementations; `training/test_features.js` checks their output parity to catch normalization drift.
- 15 sampled frames/second; each raw sign clip has variable duration.
- Each complete variable-duration sign segment is linearly resampled to 30 frames × 456 features using the same shared JavaScript routine for live inference; training uses the equivalent interpolation. The v2 resampler keeps missing values discrete rather than interpolating them into fake coordinates.
- 456 features = 132 pose values + 252 hand values + 72 selected face values. Pose points that are unavailable use visibility 0; missing hand/face coordinates use the reserved -3 sentinel and remain distinguishable from real coordinates.
- Raw camera-recorded or imported video stays in local IndexedDB. Clearing browser site data can remove it; the collector does not provide a bulk video ZIP export. Existing videos must be one sign per imported clip; the tool does not automatically detect boundaries or assign labels in continuous multi-sign videos. Dataset metadata also records that the checker marked the sign form verified and that informed recording consent was confirmed.
- No cloud upload or automatic sync occurs.


## Troubleshooting: the counter stays at zero

- **Start sample** records from the live camera. The frame counter counts landmark feature frames, not only raw video frames. While recording, the interface now shows MediaPipe result callbacks, latest visible pose/hand/face point counts, and valid feature frames.
- **Import and extract landmarks** processes an existing clip. It shows progress and usable frame counts separately.
- If the diagnostic says camera frames were sent but MediaPipe returned no callbacks, check the browser console and internet access for the MediaPipe CDN. Serve the page on `http://localhost:8001` and refresh with Ctrl+Shift+R.
- If callbacks arrive but all landmark counts are zero, improve lighting, bring the signer closer, and ensure a hand, part of the upper body, or at least some face keypoints are in frame. The toolkit does not create fake landmarks when nothing is detected.
- Partial-landmark acceptance may increase low-information samples. Review sample quality and validate on signers excluded from training.
