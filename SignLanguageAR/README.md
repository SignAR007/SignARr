# SignAR — Filipino Sign Language Decoder

A static, browser-based FSL decoder project. The current foundation tracks hands, pose/body, and selected face landmarks in the browser and normalizes those coordinates for a temporal TensorFlow.js model.

## Current state

- Camera-first public Decoder Mode.
- Rear camera first, with fallback when it is unavailable; front/rear switching is supported.
- Camera capture supports up to 1280×720. Holistic starts at 960 px with pose model complexity 1; input resolution adapts between 720, 960, and 1280 px to balance tracking detail and responsiveness.
- MediaPipe Holistic draws confidence-filtered body joints, outlined left/right hand landmarks, and a connected face oval with key eye, brow, and mouth landmarks.
- A low-resolution face detector outlines multiple visible faces and pauses sign predictions when it sees two or more. Holistic still tracks one person's body and hands at a time; it does not independently decode multiple signers.
- Landmark normalization uses a torso anchor/scale and wrist-relative hand geometry.
- Live inference buffers a complete variable-duration sign segment and uses the same linear 30-frame resampling algorithm as the Python trainer.
- Speech clears the spoken decoded sentence after playback; recognized-sign history remains visible and can still be deleted or cleared manually.
- Real model predictions are withheld below a strict 99% model-score threshold; lower-scoring results do not appear as sign labels or enter the sentence.
- TensorFlow.js loads only a validated model manifest with at least two labels; the public repository intentionally has **no validated model installed yet**. Landmark tracking still works, while model inference and its additional multi-person detector remain off until a reviewed model is available.
- No sign labels are fabricated while the trained model is absent.
- The public decoder uses Filipino-only display and prefers a browser Filipino voice, with a local browser speech fallback.
- Raw training recordings are not stored or uploaded by the public decoder.

## Run and deploy the decoder

From the project repository root, serve the decoder directory locally:

```powershell
python -m http.server 8000 --directory SignLanguageAR
```

Open `http://localhost:8000/` in a modern browser. Camera access requires HTTPS or localhost. The public decoder contains no simulated-result, sample-collection, training, or candidate-try mode; query parameters do not enable these features. If the validated model is absent, landmark tracking remains available but sign predictions are disabled.

The public decoder loads only `model/vocabulary.json` and the model assets it names. A deployable manifest must have `trained: true`, a model path, the `signar-fsl-456-v2` feature schema, 30-frame × 456-feature input, and at least two labels in the exact model output order. Input/output tensor shapes are checked before inference. A one-label model cannot distinguish signs and is rejected.

For GitHub Pages, use the repository's Pages workflow, which publishes only this directory. Set the Pages source to **GitHub Actions** in repository settings. After deployment, open the HTTPS site, verify the camera permission prompt, and confirm the page reports that no model is installed unless reviewed model assets have been added. The prototype's MediaPipe dependencies are loaded from CDNs and require an internet connection. MediaPipe Holistic tries jsDelivr and falls back to unpkg if the first CDN is unavailable. TensorFlow.js is loaded from CDN only when a compatible trained model is configured.

Model scores are not calibrated probabilities. The strict 99% score threshold can suppress correct signs and does not guarantee that a displayed prediction is correct. Do not rely on this prototype for communication or accessibility-critical decisions.

## Important model status

**This is a foundation, not yet a validated FSL decoder.** A temporal model must be evaluated using verified FSL recordings before the camera can reliably decode words. The separate training toolkit can train incomplete datasets for experimentation. The decoder reads labels from the model manifest, but a one-label model cannot distinguish signs and a model trained with a different vocabulary is not a useful translator. Have the actual forms reviewed by FSL signers, a teacher, or an interpreter before collecting labels. See the [training guide](../SignAR_FSL_Training_Tools_Video_Import_v5/TRAINING.md).

## Inference performance

MediaPipe landmark tracking and canvas drawing run for every processed camera frame; this is the main continuous workload. The preview requests 1280×720 instead of 4K, while Holistic starts at 960 px and pose complexity 1, then adapts between 720, 960, and 1280 px. If frame processing exceeds 65 ms for three consecutive frames, input resolution steps down; it steps back up only after 120 frames under 42 ms. Processing cadence also adapts between 12–30 FPS. When no validated model is installed, feature-sequence preparation and the additional multi-person detector are skipped because no predictions can be made. The processing canvas keeps its backing buffer between frames and resizes only when dimensions change. Model inference runs asynchronously at an adaptive 100–180 ms interval when a model is available.

A separate 640 px face check runs twice per second. It adds visible face boxes and pauses predictions on multi-person frames to avoid associating the single Holistic result with the wrong signer. This is a safety guard, not per-person body/hand tracking.

Batching is unlikely to improve a single live signer because each prediction contains only one sequence. Weight quantization may reduce model download size and memory, but it does not guarantee faster browser inference and can change predictions; compare latency and accuracy on target devices before adopting it. Lowering MediaPipe landmark quality settings may also change the feature distribution, so it should only be considered alongside retraining and validation.

## Model contract

The decoder validates:
- model/vocabulary.json — class order and model contract.
- model/model.json and TensorFlow.js weight shards — only after training.
- Input tensor shape [batch, 30, 456].
- Output tensor shape [batch, number-of-manifest-labels], with output order matching the manifest.

Feature layout:
- 33 pose landmarks × (x, y, z, visibility) = 132 values.
- Left and right hand landmarks, each with 21 body-relative and wrist/palm-relative points = 252 values combined.
- 24 selected face points × (x, y, z) = 72 values.
- Total = 456 values per sampled frame.

The first model is intended for right-handed signers and front-facing sign execution. Evaluation should split by signer so test participants do not appear in training. A candidate is not ready to install just because training completed: review its evaluation, verify that its labels and class order match the decoder, and confirm the sign forms with FSL experts.

## Private training data

Store raw videos in a local/private dataset folder or private backup; do not commit raw videos to the public repository. Export only validated model artifacts and necessary public metadata after review and testing.