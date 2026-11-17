(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video = $('camera');
  const canvas = $('landmarkCanvas');
  const ctx = canvas.getContext('2d');
  const startCameraBtn = $('startCamera');
  const stopCameraBtn = $('stopCamera');
  const startRecordBtn = $('startRecording');
  const stopRecordBtn = $('stopRecording');
  const signerInput = $('signerId');
  const labelSelect = $('labelSelect');
  const verifyCheck = $('verified');
  const consentCheck = $('consent');
  const notesInput = $('verificationNotes');
  const autoCheck = $('autoCapture');
  const trackStatus = $('trackingStatus');
  const status = $('storageStatus');
  const stageLabel = $('stageLabel');
  const guide = $('guide');
  const recordState = $('recordState');
  const frameCounter = $('frameCounter');
  const frameDiagnostics = $('frameDiagnostics');
  const captureHelp = $('captureHelp');
  const sampleList = $('sampleList');
  const toast = $('toast');
  const importVideoInput = $('importVideoFile');
  const importProgress = $('importProgress');
  const workflowStatus = $('workflowStatus');
  const sampleSummary = $('sampleSummary');
  const exportJsonBtn = $('exportJson');
  const deleteAllBtn = $('deleteAll');
  const trainModelBtn = $('trainModel');
  const tryModelBtn = $('tryModel');
  const trainingMessage = $('trainingMessage');
  const modelTestMessage = $('modelTestMessage');
  const trainingProgress = $('trainingProgress');
  const trainingProgressLabel = $('trainingProgressLabel');
  const trainingEpochCount = $('trainingEpochCount');
  const trainingAccuracy = $('trainingAccuracy');
  const trainingLoss = $('trainingLoss');
  const validationAccuracy = $('validationAccuracy');
  const validationLoss = $('validationLoss');
  const workflowSteps = [1,2,3,4].map(number => $('workflowStep' + number));

  const HOLISTIC_VERSION = '0.5.1675471629';
  const HOLISTIC_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@' + HOLISTIC_VERSION;
  const DB_NAME = 'SignAR-FSL-Private-Training-v2';
  const DB_VERSION = 1;
  const TRAINING_EPOCHS = 40;
  const MODEL_STORAGE_URL = 'indexeddb://signar-fsl-browser-candidate';
  const MODEL_METADATA_KEY = 'signar-fsl-browser-candidate-metadata-v1';
  const SAMPLE_MS = 1000 / 15;
  const MAX_TRACKING_DIMENSION = 1280;
  const TRACKING_DIMENSIONS = [720,960,1280];
  const MOTION_THRESHOLD = 0.0055;
  const MIN_FRAMES = 8;
  const MIN_IMPORTED_FRAMES = 2;
  const FACE_IDS = [10,152,234,454,1,4,33,133,159,145,362,263,386,374,61,291,13,14,70,300,105,334,78,308];
  const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const POSE_LINKS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
  const FACE_LINES = [[33,160,158,133,153,144,33],[362,385,387,263,373,380,362],[61,146,91,181,84,17,314,405,321,375,291],[78,191,80,81,82,13,312,311,310,415,308],[70,63,105,66,107],[300,293,334,296,336]];

  let dbPromise = null;
  let holistic = null;
  let stream = null;
  let facingMode = 'environment';
  let rafId = null;
  let processBusy = false;
  let lastFrameTime = -1;
  let lastProcessAt = 0;
  let processEveryMs = 1000 / 30;
  let trackingDimensionIndex = TRACKING_DIMENSIONS.length - 1;
  let slowTrackingFrames = 0;
  let fastTrackingFrames = 0;
  let processCanvas = document.createElement('canvas');
  let processContext = processCanvas.getContext('2d');
  let previousFeatures = null;
  let capture = null;
  let mediaRecorder = null;
  let mediaChunks = [];
  let captureMeta = null;
  let captureFrames = [];
  let lastSampleAt = 0;
  let captureStartedAt = 0;
  let lastMotionAt = 0;
  let saving = false;
  let vocabulary = [];
  let vocabularyConfig = null;
  let savedSampleCount = 0;
  let savedLabelCount = 0;
  let savedSampleIds = [];
  let training = false;
  let trainingCompleted = false;
  let modelTest = null;
  let modelTesting = false;
  let importSession = null;
  let importing = false;
  let importScheduled = false;
  let importedVideoUrl = null;
  let trackerResultsSeen = 0;
  let trackerFeaturesSeen = 0;
  let trackerFramesSent = 0;
  let trackerCurrentCounts = {pose:0,left:0,right:0,face:0};

  const safeText = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const measureDistance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
  function adaptTrackingResolution(elapsed) {
    if (elapsed > 65) {
      slowTrackingFrames++;fastTrackingFrames=0;
      if (slowTrackingFrames >= 3) {trackingDimensionIndex=Math.max(0,trackingDimensionIndex-1);slowTrackingFrames=0;}
    } else if (elapsed < 42) {
      fastTrackingFrames++;slowTrackingFrames=0;
      if (fastTrackingFrames >= 120) {trackingDimensionIndex=Math.min(TRACKING_DIMENSIONS.length-1,trackingDimensionIndex+1);fastTrackingFrames=0;}
    } else {slowTrackingFrames=0;fastTrackingFrames=0;}
  }

  function updateLabelOptions(samples = []) {
    const labels = new Map();
    [...vocabulary,...samples.map(sample => sample.label)].forEach(value => {
      const label = String(value ?? '').trim();
      const key = label.toUpperCase();
      if (label && !labels.has(key)) labels.set(key,label);
    });
    $('labelOptions').innerHTML = [...labels.values()]
      .map(label => '<option value="' + safeText(label) + '">')
      .join('');
  }

  function notify(message) {
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => toast.classList.remove('show'), 3200);
  }

  async function predictBrowserCandidate(frames) {
    const serialized=localStorage.getItem(MODEL_METADATA_KEY);
    if(!serialized)throw new Error('The saved model labels are missing. Train the model again before testing.');
    const metadata=JSON.parse(serialized);
    const labels=metadata.manifest?.labels;
    if(!Array.isArray(labels)||labels.length<1)throw new Error('The saved model has no sign labels to predict.');
    const tf=window.tf;
    if(!tf?.loadLayersModel)throw new Error('TensorFlow.js cannot load the saved browser model.');
    await tf.ready();
    const model=await tf.loadLayersModel(MODEL_STORAGE_URL);
    try {
      const outputShape=model.outputs[0]?.shape;
      if(!outputShape||outputShape[outputShape.length-1]!==labels.length)throw new Error('Saved model outputs do not match its sign labels. Train the model again.');
      return await window.SignARBrowserTrainer.predict(model,frames,labels);
    } finally {
      model.dispose();
    }
  }

  async function finishModelTest(frames) {
    modelTesting=true;
    modelTestMessage.textContent='Checking this fresh performance against the saved model…';
    updateWorkflow();
    try {
      const results=await predictBrowserCandidate(frames);
      const best=results[0];
      const alternatives=results.slice(1,3).map(result=>result.label+' ('+(result.score*100).toFixed(1)+'%)').join(', ');
      modelTestMessage.textContent='Best match: '+best.label+' ('+(best.score*100).toFixed(1)+'% model score)'+(alternatives?'. Other matches: '+alternatives:'')+'. This is a quick check, not a measured accuracy score; repeat with new performances and signers.';
    } catch(error) {
      console.error(error);
      modelTestMessage.textContent='Could not test the model: '+(error.message||error.name||'Unknown error');
      notify('Model test failed: '+(error.message||error.name||'Unknown error'));
    } finally {
      modelTesting=false;
      updateWorkflow();
    }
  }

  function startModelTest() {
    if(!trainingCompleted||!stream||modelTest||modelTesting)return;
    modelTest={frames:[],lastSampleAt:0};
    modelTestMessage.textContent='Ready. Perform one sign clearly now; up to 30 valid frames will be checked and not saved.';
    updateWorkflow();
  }

  function isSetupReady() {
    return !!signerInput.value.trim() && !!labelSelect.value.trim() &&
      notesInput.value.trim().length >= 5 && verifyCheck.checked && consentCheck.checked;
  }

  function updateWorkflow() {
    const setupReady = isSetupReady();
    const hasSamples = savedSampleCount > 0;
    const stepsComplete = [setupReady, !!stream || hasSamples, hasSamples, trainingCompleted];
    const currentStep = stepsComplete.findIndex(complete => !complete);

    workflowSteps.forEach((step, index) => {
      const complete = stepsComplete[index];
      step.classList.toggle('complete', complete);
      step.classList.toggle('current', index === currentStep);
      if (index === currentStep) step.setAttribute('aria-current', 'step');
      else step.removeAttribute('aria-current');
      step.querySelector('.step-state').textContent = complete ? 'Complete' : index === currentStep ? 'In progress' : 'Next';
    });

    if (importing) workflowStatus.textContent = 'Importing your clip locally. Keep this page open until it finishes.';
    else if (training) workflowStatus.textContent = 'Training your model on this device. You can follow each epoch below.';
    else if (capture) workflowStatus.textContent = 'Recording in progress. Perform one sign, then stop or pause to save.';
    else if (saving) workflowStatus.textContent = 'Saving your sample to this browser…';
    else if (!setupReady) workflowStatus.textContent = 'Enter a signer alias and sign label, add a reference note, then confirm verification and consent.';
    else if (!stream && !hasSamples) workflowStatus.textContent = 'Details are ready. Start the camera or choose a clip; selected clips import automatically.';
    else if (!hasSamples) workflowStatus.textContent = 'Ready to collect. Start recording, or turn on hands-free capture to start on movement.';
    else if (savedLabelCount < 1) workflowStatus.textContent = savedSampleCount + ' sample' + (savedSampleCount === 1 ? '' : 's') + ' saved. Add a sign label before training.';
    else if (!trainingCompleted) workflowStatus.textContent = savedSampleCount + ' sample' + (savedSampleCount === 1 ? '' : 's') + ' saved. Select Start Training when ready.';
    else workflowStatus.textContent = 'Training completed and the model was saved in this browser. It is not deployed or approved for deployment.';

    if (!capture && !saving && !importing && !training && !hasSamples && !captureHelp.textContent.startsWith('Sample too short')) {
      captureHelp.textContent = !setupReady
        ? 'Complete Step 1 and start the camera to record a sample.'
        : !stream
          ? 'Your sign details are ready. Start the camera to record, or import a video from Step 1.'
          : autoCheck.checked
            ? 'Hands-free capture is ready. Make sure the signer is ready; movement starts recording and a pause saves the sample.'
            : 'Your camera and sign details are ready. Press Start recording, then Stop and save when the sign is complete.';
    }

    if (importVideoInput.files?.length && !importing &&
        (importProgress.textContent.startsWith('Clip selected') || importProgress.textContent.startsWith('Ready to analyze'))) {
        importProgress.textContent = training
          ? 'Clip selected; it will import automatically when training finishes.'
          : setupReady
        ? 'Ready to analyze ' + importVideoInput.files[0].name + ' locally; import starts automatically.'
        : 'Clip selected. Complete the label, reference, verification, and consent; it will then import automatically.';
    }

    startRecordBtn.disabled = !stream || !setupReady || capture || saving || importing || training || !!modelTest || modelTesting;
    importVideoInput.disabled = capture || saving || importing || training || !!modelTest || modelTesting;
    exportJsonBtn.disabled = !hasSamples;
    deleteAllBtn.disabled = !hasSamples || importing || training || capture || saving || !!modelTest || modelTesting;
    trainModelBtn.disabled = savedLabelCount < 1 || training || capture || saving || importing || !!modelTest || modelTesting;
    trainModelBtn.textContent = trainingCompleted ? 'Train Again' : 'Start Training';
    tryModelBtn.disabled = !trainingCompleted || !stream || training || capture || saving || importing || !!modelTest || modelTesting;
    tryModelBtn.textContent = modelTest ? 'Collecting sign…' : modelTesting ? 'Checking…' : 'Try a sign';
    sampleList.querySelectorAll('[data-delete]').forEach(button => {
      button.disabled = importing || training || capture || saving || !!modelTest || modelTesting;
    });
    startCameraBtn.disabled = importing || training || !!stream;
    stopCameraBtn.disabled = importing || training || !stream;

    if (importVideoInput.files?.length && setupReady && !capture && !saving && !importing && !training && !importScheduled) {
      importScheduled = true;
      setTimeout(() => {
        importScheduled = false;
        if (importVideoInput.files?.length && isSetupReady() && !capture && !saving && !importing && !training) {
          void importVideoSample();
        }
      }, 0);
    }
  }

  function openDb() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve,reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains('samples')) database.createObjectStore('samples',{keyPath:'id'});
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB is unavailable'));
    });
    return dbPromise;
  }

  async function dbRequest(mode, operation) {
    const database = await openDb();
    return new Promise((resolve,reject) => {
      const transaction = database.transaction('samples',mode);
      const store = transaction.objectStore('samples');
      const request = operation(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Local database operation failed'));
    });
  }
  const getSamples = async () => (await dbRequest('readonly',store => store.getAll())).sort((a,b) => b.createdAt-a.createdAt);
  const saveSample = sample => dbRequest('readwrite',store => store.put(sample));
  const deleteSample = id => dbRequest('readwrite',store => store.delete(id));
  const clearSamples = () => dbRequest('readwrite',store => store.clear());

  function updateRecordState(active) {
    recordState.classList.toggle('recording',active);
    recordState.children[1].textContent = active ? (captureMeta?.auto ? 'Automatic capture running' : 'Recording sample…') : 'Not recording';
    frameCounter.textContent = captureFrames.length + ' frames';
    stopRecordBtn.disabled = !active;
    signerInput.disabled = active;
    labelSelect.disabled = active;
    verifyCheck.disabled = active;
    consentCheck.disabled = active;
    notesInput.disabled = active;
    updateWorkflow();
  }

  function validateCaptureForm() {
    if (!stream) { notify('Start the camera first.'); return false; }
    if (!signerInput.value.trim()) { notify('Enter a signer alias such as signer-01.'); return false; }
    if (!labelSelect.value) { notify('Choose an FSL label.'); return false; }
    if (!verifyCheck.checked) { notify('Verify the sign form with an FSL reference and qualified signer first.'); return false; }
    if (!consentCheck.checked) { notify('Get informed permission before recording a signer.'); return false; }
    if (notesInput.value.trim().length < 5) { notify('Add a short FSL reference/checker note before recording.'); return false; }
    if (!window.MediaRecorder) { notify('This browser does not support MediaRecorder.'); return false; }
    return true;
  }

  function mediaRecorderType() {
    if (!window.MediaRecorder?.isTypeSupported) return '';
    return ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'].find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  function startCapture(isAuto = false) {
    if (capture || saving || !validateCaptureForm()) return;
    const mimeType = mediaRecorderType();
    try {
      mediaChunks = [];
      captureFrames = [];
      lastSampleAt = 0;
      captureStartedAt = performance.now();
      lastMotionAt = captureStartedAt;
      captureMeta = {
        id: (crypto.randomUUID?.() || ('sample-' + Date.now() + '-' + Math.random().toString(36).slice(2,9))),
        signerId: signerInput.value.trim().replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,40),
        label: labelSelect.value,
        verificationNotes: notesInput.value.trim(),
        verified: true,
        consentRecorded: true,
        auto: isAuto,
        createdAt: Date.now()
      };
      mediaRecorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
      mediaRecorder.ondataavailable = event => { if (event.data?.size) mediaChunks.push(event.data); };
      mediaRecorder.onerror = event => notify('Recorder error: ' + (event.error?.message || 'unknown error'));
      mediaRecorder.onstop = finishCapture;
      mediaRecorder.start(250);
      capture = true;
      if(frameDiagnostics) frameDiagnostics.textContent = 'Recording is active. Waiting for usable landmark frames; the count below reports valid frames / analyzed frames.';
      frameCounter.textContent = '0 valid / waiting for results';
      stageLabel.textContent = isAuto ? 'AUTO RECORDING · ' + captureMeta.label : 'RECORDING · ' + captureMeta.label;
      captureHelp.textContent = isAuto ? 'Motion detected. Relax and pause to save the sample automatically.' : 'Perform the verified sign naturally. Press Stop and save when the sign is complete.';
      updateRecordState(true);
    } catch (error) {
      mediaRecorder = null; capture = null;
      notify('Unable to start recording: ' + (error.message || error.name));
    }
  }

  function stopCapture() {
    if (!capture || !mediaRecorder || mediaRecorder.state === 'inactive') return;
    capture = false;
    saving = true;
    updateRecordState(false);
    stageLabel.textContent = 'Saving sample…';
    try { mediaRecorder.stop(); }
    catch (error) { saving = false; notify('Could not stop recorder: ' + error.message); updateRecordState(false); }
  }

  async function finishCapture() {
    const meta = captureMeta;
    const frames = captureFrames.slice();
    const type = mediaRecorder?.mimeType || mediaChunks[0]?.type || 'video/webm';
    const videoBlob = new Blob(mediaChunks,{type});
    mediaRecorder = null; captureMeta = null; mediaChunks = []; captureFrames = []; capture = false;
    updateRecordState(false);
    if (!meta) { saving = false; return; }
    if (frames.length < MIN_FRAMES || videoBlob.size === 0) {
      saving = false;
      stageLabel.textContent = 'Sample not saved';
      captureHelp.textContent = 'Sample too short. Record at least ' + MIN_FRAMES + ' landmark frames and try again.';
      notify('Sample not saved: too short or no raw video was captured.');
      updateRecordState(false);
      return;
    }
    const sample = {...meta,sourceType:'camera',features:frames,frameCount:frames.length,videoBlob,videoType:type,videoExtension:type.includes('mp4')?'mp4':'webm',durationMs:Math.round(performance.now()-captureStartedAt)};
    try {
      await saveSample(sample);
      stageLabel.textContent = 'Saved · ' + sample.label;
      captureHelp.textContent = 'Saved locally. Change the label or signer alias before recording the next sample.';
      notify('Saved ' + sample.label + ' — ' + frames.length + ' landmark frames.');
      await refreshSamples();
    } catch (error) {
      stageLabel.textContent = 'Local storage error';
      notify('Could not save sample. Local browser storage may be full: ' + error.message);
    } finally {
      saving = false;
      updateRecordState(false);
    }
  }

  function canvasPoint(p,w,h) {
    const rect=video.getBoundingClientRect(),sourceW=video.videoWidth||1280,sourceH=video.videoHeight||720;
    if(!rect.width||!rect.height)return {x:p.x*w,y:p.y*h};
    const scale=Math.max(rect.width/sourceW,rect.height/sourceH),offsetX=(rect.width-sourceW*scale)/2,offsetY=(rect.height-sourceH*scale)/2;
    return {x:(p.x*sourceW*scale+offsetX)*w/rect.width,y:(p.y*sourceH*scale+offsetY)*h/rect.height};
  }
  function drawLinks(points,links,w,h,color,width=2) {
    if (!points?.length) return;
    ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=Math.max(1.3,w/600*width); ctx.lineCap='round'; ctx.lineJoin='round';
    for (const [a,b] of links) { if (!points[a]||!points[b]) continue; const pa=canvasPoint(points[a],w,h),pb=canvasPoint(points[b],w,h);ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke(); }
    ctx.restore();
  }
  function drawPoints(points,w,h,color,indices=null) {
    if (!points?.length) return;
    ctx.save();ctx.fillStyle=color;
    for (const i of (indices||points.map((_,n)=>n))) { const p=points[i]; if(!p)continue;const point=canvasPoint(p,w,h);ctx.beginPath();ctx.arc(point.x,point.y,Math.max(1.3,w/850),0,Math.PI*2);ctx.fill(); }
    ctx.restore();
  }
  function getLandmarkCounts(results) {
    const validPoint = p => !!p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.x >= -0.5 && p.x <= 1.5 && p.y >= -0.5 && p.y <= 1.5;
    const pose = results?.poseLandmarks || [], left = results?.leftHandLandmarks || [], right = results?.rightHandLandmarks || [], face = results?.faceLandmarks || [];
    const poseCount = [0,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24].filter(i => validPoint(pose[i]) && (pose[i].visibility ?? 1) >= 0.15).length;
    const count = list => (list || []).slice(0,21).filter(validPoint).length;
    const faceCount = FACE_IDS.filter(i => validPoint(face[i])).length;
    return { pose:poseCount, left:count(left), right:count(right), face:faceCount };
  }

  function updateFrameDiagnostics(results, feature) {
    trackerResultsSeen++;
    trackerCurrentCounts = getLandmarkCounts(results || {});
    if (feature) trackerFeaturesSeen++;
    if (!frameDiagnostics) return;
    const c = trackerCurrentCounts;
    const hasAny = c.pose + c.left + c.right + c.face > 0;
    const recordingText = capture ? ` · valid samples ${captureFrames.length}` : '';
    if (trackerResultsSeen === 0) return;
    frameDiagnostics.dataset.state = feature ? 'ok' : 'warning';
    frameDiagnostics.textContent = `MediaPipe results: ${trackerResultsSeen} · usable feature frames: ${trackerFeaturesSeen} · latest landmarks: pose ${c.pose}, left hand ${c.left}, right hand ${c.right}, selected face ${c.face}${recordingText}. ` +
      (feature ? 'Partial landmarks accepted.' : hasAny ? 'Some points were found but did not form a usable feature frame.' : 'No landmarks in this frame. Move closer, improve lighting, and keep at least a hand, part of the upper body, or the face in view.');
  }

  function paintLandmarks(r) {
    const sw=video.videoWidth||1280,sh=video.videoHeight||720,scale=Math.min(1,MAX_TRACKING_DIMENSION/Math.max(sw,sh)),w=Math.max(1,Math.round(sw*scale)),h=Math.max(1,Math.round(sh*scale));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}ctx.clearRect(0,0,w,h);
    const pose=r.poseLandmarks||[],left=r.leftHandLandmarks||[],right=r.rightHandLandmarks||[],face=r.faceLandmarks||[];
    drawLinks(pose,POSE_LINKS,w,h,'#5ebdff',2);drawPoints(pose,w,h,'#c8eaff',[0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]);
    drawLinks(left,HAND_LINKS,w,h,'#b8a3ff',2.7);drawPoints(left,w,h,'#e0d8ff');drawLinks(right,HAND_LINKS,w,h,'#43dfad',2.7);drawPoints(right,w,h,'#a7f3d0');
    const lines=[];for(const path of FACE_LINES)for(let i=1;i<path.length;i++)lines.push([path[i-1],path[i]]);drawLinks(face,lines,w,h,'#ffcf70',1.1);drawPoints(face,w,h,'#ffe2a6',FACE_IDS);
  }

  async function initHolistic() {
    if (holistic) return;
    if (typeof window.Holistic !== 'function') throw new Error('MediaPipe Holistic did not load. Check internet access and refresh.');
    holistic = new window.Holistic({locateFile:file => HOLISTIC_BASE + '/' + file});
    holistic.setOptions({modelComplexity:2,smoothLandmarks:true,refineFaceLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
    holistic.onResults(results => {
      paintLandmarks(results);
      const feature = window.SignARFeatureExtractor?.extract(results,video.videoWidth||1280,video.videoHeight||720) || null;
      updateFrameDiagnostics(results, feature);
      guide.classList.toggle('hidden',!!feature);
      if (importSession) {
        importSession.processedFrames++;
        const pose = results.poseLandmarks || [], face = results.faceLandmarks || [];
        const validPoint = p => !!p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.x >= -0.5 && p.x <= 1.5 && p.y >= -0.5 && p.y <= 1.5;
        const poseRelevant = [0,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
          .filter(i => validPoint(pose[i]) && (pose[i].visibility ?? 1) >= .15);
        const handVisible = hand => Array.isArray(hand) && hand.slice(0,21).filter(validPoint).length >= 3;
        const leftHandVisible = handVisible(results.leftHandLandmarks || []);
        const rightHandVisible = handVisible(results.rightHandLandmarks || []);
        const faceVisible = FACE_IDS.filter(i => validPoint(face[i])).length >= 3;
        const shoulders = pose.length >= 33 && [11,12].every(i => validPoint(pose[i]) && (pose[i].visibility ?? 1) >= .25);
        if (poseRelevant.length >= 2) importSession.poseFrames++;
        if (shoulders) importSession.shoulderFrames++;
        if (pose.length >= 33 && [23,24].every(i => validPoint(pose[i]) && (pose[i].visibility ?? 1) >= .25)) importSession.hipFrames++;
        if (leftHandVisible) importSession.leftHandFrames++;
        if (rightHandVisible) importSession.rightHandFrames++;
        if (faceVisible) importSession.faceFrames++;
        if (feature && !shoulders) importSession.fallbackFrames++;
        if (feature) importSession.frames.push(feature);
        frameCounter.textContent = importSession.frames.length + ' valid frames';
        trackStatus.textContent = feature ? 'Imported signer tracked' : 'Signer not detected in this frame';
        return;
      }
      trackStatus.textContent = feature ? 'Landmarks usable' : 'No usable landmarks in latest frame';
      if (!feature) {
        previousFeatures=null;
        if (capture) frameCounter.textContent = `${captureFrames.length} valid / ${trackerResultsSeen} analyzed`;
        if (modelTest) modelTestMessage.textContent='No usable landmarks yet. Move closer and keep a hand or your upper body visible.';
        return;
      }
      const now=performance.now();let energy=0;
      if(modelTest){
        if(now-modelTest.lastSampleAt>=SAMPLE_MS){
          modelTest.frames.push(feature);
          modelTest.lastSampleAt=now;
          modelTestMessage.textContent='Capturing a fresh sign locally: '+modelTest.frames.length+' of 30 frames. Keep signing clearly for about two seconds.';
          if(modelTest.frames.length>=30){
            const frames=modelTest.frames;
            modelTest=null;
            void finishModelTest(frames);
          }
        }
        return;
      }
      if(previousFeatures){for(let i=0;i<456;i++)energy+=Math.abs(feature[i]-previousFeatures[i]);energy/=456;}
      previousFeatures=feature;
      if(energy>=MOTION_THRESHOLD){lastMotionAt=now;if(autoCheck.checked&&verifyCheck.checked&&consentCheck.checked&&labelSelect.value&&signerInput.value.trim()&&notesInput.value.trim().length>=5&&!capture&&!saving&&!modelTesting)startCapture(true);}
      if(capture&&now-lastSampleAt>=SAMPLE_MS){captureFrames.push(feature);lastSampleAt=now;frameCounter.textContent=`${captureFrames.length} valid / ${trackerResultsSeen} analyzed`;}
      if(autoCheck.checked&&capture&&now-lastMotionAt>850&&captureFrames.length>=MIN_FRAMES&&now-captureStartedAt>1000)stopCapture();
    });
    if (typeof holistic.initialize === 'function') await holistic.initialize();
  }

  function validImportForm(file) {
    if (!file) { notify('Choose a recorded video first.'); return false; }
    if (!file.type.startsWith('video/') && !/\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name)) {
      notify('Choose a video file such as MP4 or WebM.'); return false;
    }
    if (!signerInput.value.trim()) { notify('Enter a signer alias such as signer-01.'); return false; }
    if (!labelSelect.value) { notify('Choose the FSL label performed in this clip.'); return false; }
    if (!verifyCheck.checked) { notify('Verify this exact sign form with a reliable FSL reference and qualified signer first.'); return false; }
    if (!consentCheck.checked) { notify('Confirm that the signer gave permission for local storage and analysis.'); return false; }
    if (notesInput.value.trim().length < 5) { notify('Add a reference/checker note of at least five characters.'); return false; }
    if (capture || saving || importing) { notify('Finish the current recording or import first.'); return false; }
    return true;
  }

  function waitForMetadata() {
    if (video.readyState >= 1 && Number.isFinite(video.duration)) return Promise.resolve();
    return new Promise((resolve,reject) => {
      const timer=setTimeout(()=>{cleanup();reject(new Error('Timed out reading this video. Try MP4 or WebM.'));},12000);
      const loaded=()=>{cleanup();resolve();};
      const failed=()=>{cleanup();reject(new Error('This browser cannot decode the selected video. Try MP4 (H.264) or WebM.'));};
      function cleanup(){clearTimeout(timer);video.removeEventListener('loadedmetadata',loaded);video.removeEventListener('error',failed);}
      video.addEventListener('loadedmetadata',loaded,{once:true});video.addEventListener('error',failed,{once:true});
    });
  }

  function seekImportedVideo(time) {
    const target=Math.max(0,Math.min(time,Math.max(0,(video.duration||0)-0.01)));
    if (video.readyState >= 2 && Math.abs(video.currentTime-target)<0.002) return Promise.resolve();
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{cleanup();reject(new Error('The browser could not seek through this video. Try converting it to MP4 or WebM.'));},8000);
      const done=()=>{cleanup();resolve();};
      const failed=()=>{cleanup();reject(new Error('A video frame could not be decoded.'))};
      function cleanup(){clearTimeout(timer);video.removeEventListener('seeked',done);video.removeEventListener('error',failed);}
      video.addEventListener('seeked',done,{once:true});video.addEventListener('error',failed,{once:true});
      try{video.currentTime=target;}catch(error){cleanup();reject(error);}
    });
  }

  async function importVideoSample() {
    const file=importVideoInput.files?.[0];
    if(!validImportForm(file))return;
    importing=true;
    [signerInput,labelSelect,verifyCheck,consentCheck,notesInput,importVideoInput].forEach(input=>input.disabled=true);
    updateWorkflow();
    startCameraBtn.disabled=true;stopCameraBtn.disabled=true;startRecordBtn.disabled=true;stopRecordBtn.disabled=true;
    importProgress.textContent='Preparing local video…';status.textContent='Importing video…';
    try {
      if (stream) await stopCamera();
      while (processBusy) await new Promise(resolve=>setTimeout(resolve,25));
      await initHolistic();
      video.pause();video.srcObject=null;video.controls=true;video.muted=true;video.loop=false;
      importedVideoUrl=URL.createObjectURL(file);video.src=importedVideoUrl;video.classList.add('active');video.load();
      await waitForMetadata();
      if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error('This video is empty or has no readable duration.');
      video.pause();await seekImportedVideo(0);
      const duration=video.duration;
      const steps=Math.max(MIN_IMPORTED_FRAMES,Math.ceil(duration/SAMPLE_MS));
      const extension=(file.name.match(/\.(mp4|webm|mov|m4v|ogv)$/i)||[])[1]?.toLowerCase() || (file.type.includes('mp4')?'mp4':file.type.includes('quicktime')?'mov':'webm');
      const mime=file.type || (extension==='mp4'?'video/mp4':extension==='mov'?'video/quicktime':'video/webm');
      const session={frames:[],processedFrames:0,poseFrames:0,shoulderFrames:0,hipFrames:0,leftHandFrames:0,rightHandFrames:0,faceFrames:0,fallbackFrames:0};importSession=session;
      stageLabel.textContent='IMPORTING · '+labelSelect.value;
      trackStatus.textContent='Analyzing recorded video';
      frameCounter.textContent='0 valid frames';
      guide.classList.remove('hidden');
      const interval=SAMPLE_MS;
      for(let index=0;index<steps;index++){
        if(!importing)throw new Error('Import cancelled.');
        const time=Math.min(index*interval,Math.max(0,duration-0.01));
        await seekImportedVideo(time);
        const frameStarted=performance.now(),sw=video.videoWidth||1280,sh=video.videoHeight||720,scale=Math.min(1,TRACKING_DIMENSIONS[trackingDimensionIndex]/Math.max(sw,sh)),width=Math.max(1,Math.round(sw*scale)),height=Math.max(1,Math.round(sh*scale));
        if(processCanvas.width!==width)processCanvas.width=width;if(processCanvas.height!==height)processCanvas.height=height;
        processContext.drawImage(video,0,0,width,height);
        await holistic.send({image:processCanvas});
        adaptTrackingResolution(performance.now()-frameStarted);
        const completed=index+1,percent=Math.min(100,Math.round(completed/steps*100));
        if(completed%5===0||completed===steps){importProgress.textContent='Analyzing frame '+completed+' of '+steps+' ('+percent+'%). Valid landmark frames: '+session.frames.length+'.';status.textContent='Import '+percent+'%';}
      }
      const frames=session.frames.slice();
      importSession=null;
      if(frames.length<MIN_IMPORTED_FRAMES){
        throw new Error('Only '+frames.length+' usable landmark frames were found (at least '+MIN_IMPORTED_FRAMES+' are needed). MediaPipe processed '+session.processedFrames+' frames; usable upper-body pose in '+session.poseFrames+', both shoulders in '+session.shoulderFrames+', hips in '+session.hipFrames+', left hand in '+session.leftHandFrames+', right hand in '+session.rightHandFrames+', selected face landmarks in '+session.faceFrames+', and fallback-normalized frames in '+session.fallbackFrames+'. Full-body, both-hand, and complete-face visibility are NOT required: at least one meaningful region (some upper-body pose, one visible hand, or several selected face points) must be detected. If every count is zero, check the video codec and MediaPipe CDN/network in the browser console. Try an MP4 encoded with H.264 or WebM.');
      }
      const sample={
        id:(crypto.randomUUID?.()||('sample-'+Date.now()+'-'+Math.random().toString(36).slice(2,9))),
        signerId:signerInput.value.trim().replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,40),
        label:labelSelect.value,verificationNotes:notesInput.value.trim(),verified:true,consentRecorded:true,
        createdAt:Date.now(),sourceType:'imported-video',features:frames,frameCount:frames.length,
        videoBlob:file.slice(0,file.size,mime),videoType:mime,videoExtension:extension,durationMs:Math.round(duration*1000)
      };
      await saveSample(sample);await refreshSamples();
      stageLabel.textContent='Imported · '+sample.label;
      importProgress.textContent='Saved '+sample.label+' from '+file.name+' ('+frames.length+' usable landmark frames and source video).';
      notify('Imported '+sample.label+' and saved the source video and '+frames.length+' landmark frames locally.');
    } catch(error) {
      console.error(error);
      importProgress.textContent='Import failed: '+(error.message||error.name||'Unknown error');
      notify('Video import failed: '+(error.message||error.name||'Unknown error'));
    } finally {
      importSession=null;importing=false;
      video.pause();video.removeAttribute('src');video.controls=false;video.load();video.classList.remove('active');
      if(importedVideoUrl){URL.revokeObjectURL(importedVideoUrl);importedVideoUrl=null;}
      ctx.clearRect(0,0,canvas.width,canvas.height);guide.classList.remove('hidden');stageLabel.textContent='Camera is off';
      importVideoInput.value='';
      [signerInput,labelSelect,verifyCheck,consentCheck,notesInput,importVideoInput].forEach(input=>input.disabled=false);
      startCameraBtn.disabled=false;stopCameraBtn.disabled=true;
      status.textContent='Local database ready';trackStatus.textContent='Not started';
      updateWorkflow();
    }
  }

  async function openCamera() {
    const constraints={width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:60}};
    try{return await navigator.mediaDevices.getUserMedia({video:{...constraints,facingMode:{exact:facingMode}},audio:false});}
    catch(error){if(['OverconstrainedError','NotFoundError','TypeError'].includes(error?.name)){try{return await navigator.mediaDevices.getUserMedia({video:constraints,audio:false});}catch{return await navigator.mediaDevices.getUserMedia({video:true,audio:false});}}throw error;}
  }

  async function startCamera() {
    if(importing){notify('Wait for the imported video to finish.');return;}
    try {
      if(!window.isSecureContext)throw new Error('Use HTTPS or http://localhost to access the camera.');
      if(!navigator.mediaDevices?.getUserMedia)throw new Error('This browser does not provide camera access.');
      startCameraBtn.disabled=true;status.textContent='Loading models…';await initHolistic();trackerResultsSeen=0;trackerFeaturesSeen=0;trackerFramesSent=0;trackerCurrentCounts={pose:0,left:0,right:0,face:0};if(frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent='Camera started; waiting for MediaPipe results.';}video.pause();video.removeAttribute('src');video.controls=false;video.load();stream=await openCamera();video.srcObject=stream;await video.play();video.classList.add('active');
      const actual=stream.getVideoTracks()[0]?.getSettings?.().facingMode;if(actual)facingMode=actual;
      startCameraBtn.disabled=true;stopCameraBtn.disabled=false;updateWorkflow();trackStatus.textContent='Camera ready';stageLabel.textContent='Align face, torso, and hands';guide.classList.remove('hidden');
      rafId=requestAnimationFrame(loop);status.textContent='Local database ready';notify('Camera ready. Landmark capture will be stored locally.');
    }catch(error){console.error(error);if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;startCameraBtn.disabled=false;stopCameraBtn.disabled=true;status.textContent='Camera unavailable';notify(error.message||error.name||'Unable to start camera');updateWorkflow();}
  }

  async function loop(time) {
    if(!stream)return;
    if(!processBusy&&video.readyState>=2&&video.currentTime!==lastFrameTime&&time-lastProcessAt>=processEveryMs){
      processBusy=true;lastFrameTime=video.currentTime;lastProcessAt=time;const started=performance.now();
      try{const sw=video.videoWidth||1280,sh=video.videoHeight||720,s=Math.min(1,TRACKING_DIMENSIONS[trackingDimensionIndex]/Math.max(sw,sh)),width=Math.max(1,Math.round(sw*s)),height=Math.max(1,Math.round(sh*s));if(processCanvas.width!==width)processCanvas.width=width;if(processCanvas.height!==height)processCanvas.height=height;processContext.drawImage(video,0,0,width,height);await holistic.send({image:processCanvas});trackerFramesSent++;if(trackerFramesSent>=10&&trackerResultsSeen===0&&frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent=`Camera frames sent to MediaPipe: ${trackerFramesSent}, but no result callbacks have arrived. Check DevTools Console and internet access to the MediaPipe CDN, then refresh.`;}}
      catch(error){console.error(error);trackStatus.textContent='Tracking error';if(frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent='MediaPipe error: '+(error.message||error.name||'unknown')+'. Check internet/CDN and browser console.';}}
      finally{const elapsed=performance.now()-started;adaptTrackingResolution(elapsed);processEveryMs=elapsed>75?1000/12:elapsed>48?1000/18:1000/30;processBusy=false;}
    }
    rafId=requestAnimationFrame(loop);
  }

  async function stopCamera() {
    if(modelTest){
      modelTest=null;
      modelTestMessage.textContent='Sign test canceled because the camera was turned off.';
    }
    if(capture)stopCapture();
    if(rafId)cancelAnimationFrame(rafId);rafId=null;
    if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;video.pause();video.srcObject=null;video.removeAttribute('src');video.controls=false;video.load();video.classList.remove('active');
    ctx.clearRect(0,0,canvas.width,canvas.height);startCameraBtn.disabled=false;stopCameraBtn.disabled=true;trackStatus.textContent='Not started';stageLabel.textContent='Camera is off';guide.classList.remove('hidden');previousFeatures=null;status.textContent='Local database ready';updateWorkflow();
  }

  function downloadBlob(blob,name) {
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
  }
  async function exportJson() {
    const samples=await getSamples();if(!samples.length){notify('No samples to export yet.');return;}
    const dataset=window.SignARBrowserTrainer.createDatasetDocument(samples,vocabulary);
    const blob=new Blob([JSON.stringify(dataset,null,2)],{type:'application/json'});downloadBlob(blob,'signar-fsl-dataset-'+new Date().toISOString().slice(0,10)+'.json');notify('Landmark JSON exported.');
  }
  function displayMetric(value, percent = false) {
    if (!Number.isFinite(value)) return '—';
    return percent ? (value * 100).toFixed(1) + '%' : value.toFixed(4);
  }

  async function removeBrowserCandidate() {
    if (!window.tf?.io?.listModels) throw new Error('TensorFlow.js browser storage is unavailable.');
    const models=await window.tf.io.listModels();
    if(models[MODEL_STORAGE_URL])await window.tf.io.removeModel(MODEL_STORAGE_URL);
    localStorage.removeItem(MODEL_METADATA_KEY);
  }

  async function verifyStoredCandidate(expectedClasses) {
    if (!window.tf?.loadLayersModel) throw new Error('TensorFlow.js cannot reload the saved model.');
    const storedModel=await window.tf.loadLayersModel(MODEL_STORAGE_URL);
    try {
      const outputShape=storedModel.outputs[0]?.shape;
      if(!outputShape||outputShape[outputShape.length-1]!==expectedClasses) {
        throw new Error('The saved model output does not match its sign labels.');
      }
    } finally {
      storedModel.dispose();
    }
  }

  async function saveBrowserCandidate(model, metadata) {
    const tf=window.tf;
    if (!tf?.io?.listModels) throw new Error('This browser does not support persistent TensorFlow.js model storage.');
    const models=await tf.io.listModels();
    if(models[MODEL_STORAGE_URL])await tf.io.removeModel(MODEL_STORAGE_URL);
    await model.save(MODEL_STORAGE_URL);
    try {
      localStorage.setItem(MODEL_METADATA_KEY,JSON.stringify(metadata));
      await verifyStoredCandidate(metadata.manifest.outputClasses);
    } catch(error) {
      const savedModels=await tf.io.listModels();
      if(savedModels[MODEL_STORAGE_URL])await tf.io.removeModel(MODEL_STORAGE_URL);
      localStorage.removeItem(MODEL_METADATA_KEY);
      throw new Error('The saved model could not be verified in this browser: '+error.message);
    }
  }

  async function restoreBrowserCandidate(samples) {
    const serialized=localStorage.getItem(MODEL_METADATA_KEY);
    if(!serialized)return false;
    const metadata=JSON.parse(serialized);
    const models=await window.tf.io.listModels();
    if(!models[MODEL_STORAGE_URL]) {
      localStorage.removeItem(MODEL_METADATA_KEY);
      return false;
    }
    const currentIds=samples.map(sample=>sample.id).sort();
    if(JSON.stringify(metadata.sampleIds)!==JSON.stringify(currentIds)) {
      await removeBrowserCandidate();
      return false;
    }
    if((metadata.manifest?.outputClasses||0)<1) {
      await removeBrowserCandidate();
      return false;
    }
    try {
      await verifyStoredCandidate(metadata.manifest.outputClasses);
    } catch(error) {
      console.error(error);
      await removeBrowserCandidate();
      trainingMessage.dataset.state='error';
      trainingMessage.textContent='The saved model could not be reloaded. Select Start Training to create a fresh candidate.';
      return false;
    }
    trainingCompleted=true;
    trainingProgress.value=100;
    trainingProgressLabel.textContent='Training complete · model verified';
    trainingEpochCount.textContent=TRAINING_EPOCHS+' / '+TRAINING_EPOCHS+' epochs';
    trainingAccuracy.textContent=displayMetric(metadata.metrics?.accuracy,true);
    trainingLoss.textContent=displayMetric(metadata.metrics?.loss);
    validationAccuracy.textContent=displayMetric(metadata.metrics?.validationAccuracy,true);
    validationLoss.textContent=displayMetric(metadata.metrics?.validationLoss);
    trainingMessage.dataset.state='success';
    trainingMessage.textContent='Training completed, and the saved model was reloaded successfully. It remains a candidate, not deployment-ready.';
    return true;
  }

  async function trainModel() {
    if (training || importing || capture || saving || !savedSampleCount) return;
    if(savedLabelCount<1) {
      trainingMessage.dataset.state='';
      trainingMessage.textContent='Save at least one labeled sign sample before training.';
      trainingProgressLabel.textContent='Waiting for a labeled sample';
      trainingEpochCount.textContent='0 / '+TRAINING_EPOCHS+' epochs';
      return;
    }
    if (!window.SignARBrowserTrainer) {
      trainingMessage.dataset.state='error';
      trainingMessage.textContent='The browser training module could not load. Refresh the page and check the local server files.';
      return;
    }
    if (!window.tf) {
      trainingMessage.dataset.state='error';
      trainingMessage.textContent='TensorFlow.js could not load. Check your internet connection, then refresh this page.';
      return;
    }

    const epochs=TRAINING_EPOCHS;
    training=true;
    trainingCompleted=false;
    trainingProgress.value=0;
    trainingProgressLabel.textContent='Preparing saved samples';
    trainingEpochCount.textContent='0 / '+epochs+' epochs';
    trainingAccuracy.textContent='—';
    trainingLoss.textContent='—';
    validationAccuracy.textContent='—';
    validationLoss.textContent='—';
    trainingMessage.dataset.state='';
    trainingMessage.textContent='Preparing landmark sequences and loading the in-browser training engine…';
    updateWorkflow();

    let result=null;
    try {
      if(stream)await stopCamera();
      const samples=await getSamples();
      const tf=window.tf;
      await tf.ready();
      trainingMessage.textContent='Training locally with the '+tf.getBackend()+' browser backend. This may take a few minutes.';
      trainingProgressLabel.textContent='Training locally';
      result=await window.SignARBrowserTrainer.train(samples,vocabulary,{
        epochs,
        onEpoch:async metrics=>{
          trainingProgress.value=metrics.epoch/metrics.epochs*100;
          trainingProgressLabel.textContent='Training epoch '+metrics.epoch+' of '+metrics.epochs;
          trainingEpochCount.textContent=metrics.epoch+' / '+metrics.epochs+' epochs';
          trainingAccuracy.textContent=displayMetric(metrics.accuracy,true);
          trainingLoss.textContent=displayMetric(metrics.loss);
          validationAccuracy.textContent=metrics.singleLabel?'—':displayMetric(metrics.validationAccuracy,true);
          validationLoss.textContent=displayMetric(metrics.validationLoss);
          await new Promise(resolve=>requestAnimationFrame(resolve));
        }
      });

      trainingProgressLabel.textContent='Saving model in this browser';
      trainingMessage.textContent='Training finished. Saving the model and its labels on this device…';
      const evaluation={
        testMode:false,
        browserTraining:true,
        evaluationMeaningful:false,
        deploymentEligible:false,
        signerIndependent:false,
        testAccuracy:null,
        validationAccuracy:result.singleLabel?null:(result.evaluation?.accuracy??result.metrics.validationAccuracy),
        trainingAccuracy:result.metrics.accuracy,
        minimumAccuracy:0.75,
        minimumClassRecall:null,
        requiredClassRecall:0.40,
        labels:result.labels,
        confusionMatrix:result.evaluation?.confusionMatrix||[],
        validationSampleCount:result.validationCount,
        sampleCounts:{train:result.trainCount,validation:result.validationCount},
        invalidSamples:result.invalidSamples.length,
        note:'Browser training uses a sample-level validation split. It does not evaluate unseen signers and is not eligible for deployment.'
      };
      const manifest={
        ...vocabularyConfig,
        trained:false,
        testMode:false,
        browserTraining:true,
        modelPath:MODEL_STORAGE_URL,
        sequenceLength:30,
        samplingFps:15,
        featureDimension:456,
        outputClasses:result.labels.length,
        labels:result.labels,
        evaluation:{
          signerIndependent:false,
          evaluationMeaningful:false,
          deploymentEligible:false,
          testAccuracy:null,
          validationAccuracy:evaluation.validationAccuracy,
          minimumAccuracy:0.75,
          minimumClassRecall:null,
          requiredClassRecall:0.40
        },
        note:result.labels.length===1
          ? 'Browser-trained candidate with one label. It cannot distinguish signs and is not eligible for deployment.'
          : 'Browser-trained candidate. Review labels and run signer-independent evaluation before considering deployment.'
      };
      const metadata={
        manifest,
        evaluation,
        metrics:result.metrics,
        sampleIds:samples.map(sample=>sample.id).sort(),
        savedAt:new Date().toISOString()
      };
      await saveBrowserCandidate(result.model,metadata);
      trainingCompleted=true;
      trainingProgress.value=100;
      trainingProgressLabel.textContent='Training complete · model verified';
      trainingMessage.dataset.state='success';
      trainingMessage.textContent=result.labels.length===1
        ? 'Model candidate saved in this browser. It contains one label and cannot distinguish signs; add sign classes and run signer-independent evaluation before deployment.'
        : 'Training completed: all 40 epochs finished, and the saved model reloaded successfully. Metrics are shown above; this candidate still needs signer-independent evaluation before deployment.';
      notify('Training finished. The model candidate is saved on this device.');
    } catch(error) {
      console.error(error);
      trainingCompleted=false;
      trainingMessage.dataset.state='error';
      trainingMessage.textContent='Training could not finish: '+(error.message||error.name||'Unknown error');
      trainingProgressLabel.textContent='Training failed';
      notify('Browser training failed: '+(error.message||error.name||'Unknown error'));
    } finally {
      if(result?.model)result.model.dispose();
      training=false;
      updateWorkflow();
    }
  }

  async function refreshSamples() {
    const samples=await getSamples();
    updateLabelOptions(samples);
    const sampleIds=samples.map(sample=>sample.id).sort();
    const labelCount=new Set(samples.map(sample=>String(sample.label||'').trim().toUpperCase()).filter(Boolean)).size;
    const samplesChanged=JSON.stringify(sampleIds)!==JSON.stringify(savedSampleIds);
    let candidateRemovalError=null;
    if(samplesChanged&&trainingCompleted) {
      try { await removeBrowserCandidate(); }
      catch(error) {
        console.error(error);
        candidateRemovalError=error;
      }
    }
    if(samplesChanged){
      trainingCompleted=false;
      trainingMessage.dataset.state='';
      trainingMessage.textContent=samples.length&&labelCount<1
        ? 'Samples are saved locally. Add a sign label before training.'
        : samples.length
          ? 'Your saved samples changed. Select Start Training when you are ready to include the latest data.'
        : 'Save at least one labeled sample, then select Start Training when you are ready. A single label can only score that sign; it cannot distinguish other signs.';
      trainingProgress.value=0;
      trainingProgressLabel.textContent=samples.length&&labelCount<1?'Waiting for a labeled sample':samples.length?'Ready to start training':'Waiting for you to start';
      trainingEpochCount.textContent='0 / '+TRAINING_EPOCHS+' epochs';
      trainingAccuracy.textContent='—';
      trainingLoss.textContent='—';
      validationAccuracy.textContent='—';
      validationLoss.textContent='—';
      if(candidateRemovalError) {
        trainingMessage.dataset.state='error';
        trainingMessage.textContent='Samples changed, but the previous candidate could not be removed from this browser: '+candidateRemovalError.message;
      }
    }
    savedSampleCount=samples.length;
    savedLabelCount=labelCount;
    savedSampleIds=sampleIds;
    sampleSummary.textContent=samples.length
      ? samples.length+' sample'+(samples.length===1?'':'s')+' saved on this device. Start training when you are ready; new samples require a new run.'
      : 'No samples yet. Each recording is saved only in this browser.';
    updateWorkflow();
    if(!samples.length){sampleList.innerHTML='<p class="empty">No samples yet. Saved samples will appear here.</p>';return;}
    sampleList.innerHTML=samples.map(s=>'<article class="sample"><div class="sample-main"><strong>'+safeText(s.label)+' <span style="color:#8baeff">·</span> '+safeText(s.signerId)+'</strong><small>'+new Date(s.createdAt).toLocaleString()+'<br>'+s.frameCount+' frames · '+(s.videoBlob?Math.max(.01,s.videoBlob.size/1048576).toFixed(1)+' MB video':'no video')+' · '+(s.verified?'verified form':'unverified')+'</small></div><div class="sample-actions"><button data-download="'+safeText(s.id)+'">Video</button><button data-delete="'+safeText(s.id)+'">Delete</button></div></article>').join('');
    sampleList.querySelectorAll('[data-download]').forEach(btn=>btn.addEventListener('click',()=>{const sample=samples.find(s=>s.id===btn.dataset.download);if(sample?.videoBlob)downloadBlob(sample.videoBlob,sample.signerId+'_'+sample.label+'_'+sample.id+'.'+(sample.videoExtension||'webm'))}));
    sampleList.querySelectorAll('[data-delete]').forEach(btn=>{
      btn.disabled=importing||training||capture||saving;
      btn.addEventListener('click',async()=>{
        if(!confirm('Delete this local sample and its video?'))return;
        await deleteSample(btn.dataset.delete);
        await refreshSamples();
        notify('Sample deleted from this browser.');
      });
    });
  }

  async function initialize() {
    try {
      const response=await fetch('../model/vocabulary.json');
      const config=await response.json();
      vocabularyConfig=config;
      vocabulary=config.labels||[];
      updateLabelOptions();
      await openDb();status.textContent='Local database ready';await refreshSamples();
      const samples=await getSamples();
      const candidateRestored=await restoreBrowserCandidate(samples);
      if(!candidateRestored&&samples.length&&trainingMessage.dataset.state!=='error') {
        trainingMessage.dataset.state='';
        trainingMessage.textContent=savedLabelCount<1
          ? 'Samples are saved locally. Add a sign label before training.'
          : 'Samples are ready. Select Start Training when you want to begin.';
        trainingProgressLabel.textContent=savedLabelCount<1?'Waiting for a labeled sample':'Ready to start training';
      }
      updateWorkflow();
    }catch(error){status.textContent='Storage or vocabulary error';notify('Could not initialize local dataset: '+error.message);}
  }

  startCameraBtn.addEventListener('click',startCamera);
  stopCameraBtn.addEventListener('click',stopCamera);
  startRecordBtn.addEventListener('click',()=>startCapture(false));
  stopRecordBtn.addEventListener('click',stopCapture);
  [signerInput,labelSelect,notesInput].forEach(input=>input.addEventListener('input',updateWorkflow));
  labelSelect.addEventListener('change',updateWorkflow);
  [verifyCheck,consentCheck].forEach(input=>input.addEventListener('change',updateWorkflow));
  importVideoInput.addEventListener('change',()=>{
    const file=importVideoInput.files?.[0];
    if(file&&!file.type.startsWith('video/')&&!/\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name)) {
      importProgress.textContent='Choose a video file such as MP4 or WebM.';
      importVideoInput.value='';
      notify('Choose a video file such as MP4 or WebM.');
      updateWorkflow();
      return;
    }
    if(file)importProgress.textContent=isSetupReady()
      ? 'Ready to analyze '+file.name+' locally; import starts automatically.'
      : 'Clip selected. Complete the label, reference, verification, and consent; it will then import automatically.';
    else importProgress.textContent='Choose a clip. If sign details or consent are missing, it will wait until they are complete.';
    updateWorkflow();
  });
  autoCheck.addEventListener('change',()=>{
    captureHelp.textContent=autoCheck.checked
      ? 'Hands-free mode starts when movement is detected and saves after a short pause. Make sure the signer is ready before enabling it.'
      : 'Manual mode: press Start recording, then Stop and save when the sign is complete.';
    updateWorkflow();
  });
  $('exportJson').addEventListener('click',exportJson);
  trainModelBtn.addEventListener('click',trainModel);
  tryModelBtn.addEventListener('click',startModelTest);
  deleteAllBtn.addEventListener('click',async()=>{if(!confirm('Permanently delete all SignAR sample videos and landmarks from this browser?'))return;await stopCamera();await clearSamples();await refreshSamples();notify('All local samples deleted.');});
  window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(track=>track.stop());});
  updateRecordState(false);
  initialize();
})();
