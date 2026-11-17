/* SignAR FSL decoder foundation. The decoder never invents model predictions.
   Model input contract: 30 frames × 456 normalized features; output classes in model/vocabulary.json. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video=$('video'), handCanvas=$('handCanvas'), handCtx=handCanvas.getContext('2d');
  const poseCanvas=$('poseCanvas'), poseCtx=poseCanvas?.getContext('2d');
  const processCanvas=document.createElement('canvas'), processCtx=processCanvas.getContext('2d');
  const personCheckCanvas=document.createElement('canvas'), personCheckCtx=personCheckCanvas.getContext('2d');
  const cameraStage=$('cameraStage'), placeholder=$('cameraPlaceholder'), status=$('cameraStatus');
  const liveBadge=$('liveBadge'), trackingBadge=$('trackingBadge'), guide=$('positionGuide');
  const startBtn=$('startCameraBtn'), stopBtn=$('stopCameraBtn'), switchBtn=$('switchCameraBtn');
  const fullscreenBtn=$('fullscreenBtn'), installBtn=$('installBtn'), detected=$('detectedSign');
  const confidenceValue=$('confidenceValue'), confidenceBar=$('confidenceBar'), translation=$('translationText');
  const typePill=$('typePill'), state=$('recognitionState'), toast=$('toast'), overlay=$('arOverlay');
  const arSign=$('arSign'), arMeaning=$('arMeaning'), rawOutput=$('rawOutput'), sentenceOutput=$('sentenceOutput');
  const speakBtn=$('speakBtn'), deleteBtn=$('deleteBtn'), clearBtn=$('clearSentenceBtn');
  const autoSpeak=$('autoSpeakToggle'), modelStatus=$('modelStatus');
  const navButtons=[...document.querySelectorAll('.nav-btn')], sections=[...document.querySelectorAll('.page-section')];

  const HOLISTIC_VERSION='0.5.1675471629';
  const HOLISTIC_BASES=['https://cdn.jsdelivr.net/npm/@mediapipe/holistic@'+HOLISTIC_VERSION,'https://unpkg.com/@mediapipe/holistic@'+HOLISTIC_VERSION];
  const FACE_DETECTION_VERSION='0.4.1646425229';
  const FACE_DETECTION_BASES=['https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@'+FACE_DETECTION_VERSION,'https://unpkg.com/@mediapipe/face_detection@'+FACE_DETECTION_VERSION];
  const TENSORFLOW_VERSION='4.22.0';
  const TENSORFLOW_BASES=['https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@'+TENSORFLOW_VERSION+'/dist/tf.min.js','https://unpkg.com/@tensorflow/tfjs@'+TENSORFLOW_VERSION+'/dist/tf.min.js'];
  const SEQUENCE_LENGTH=30, FEATURE_DIMENSION=456, SEQUENCE_SAMPLE_MS=1000/15, MIN_SEGMENT_FRAMES=8, QUIET_TO_END_MS=750, MAX_SEGMENT_FRAMES=180, MAX_TRACKING_DIMENSION=1280;
  const TRACKING_DIMENSIONS=[720,960,1280];
  const PERSON_CHECK_DIMENSION=640, PERSON_CHECK_INTERVAL_MS=500;
  const MIN_CONFIDENCE_PERCENT=99;
  // Facial points emphasize mouth, brows, eyes and face outline for non-manual FSL features.
  const FACE_IDS=[10,152,234,454,1,4,33,133,159,145,362,263,386,374,61,291,13,14,70,300,105,334,78,308];
  const POSE_IDS=[0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32];
  const HAND_LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const POSE_LINKS=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
  const FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
  const FACE_LINES=[[33,160,158,133,153,144,33],[362,385,387,263,373,380,362],[61,146,91,181,84,17,314,405,321,375,291],[78,191,80,81,82,13,312,311,310,415,308],[70,63,105,66,107],[300,293,334,296,336]];
  const FACE_OVAL_LINKS=FACE_OVAL.slice(1).map((point,index)=>[FACE_OVAL[index],point]);
  const FACE_LINKS=FACE_LINES.flatMap(line=>line.slice(1).map((point,index)=>[line[index],point]));

  let stream=null, holistic=null, holisticBase=HOLISTIC_BASES[0], holisticLoadPromise=null, faceDetection=null, faceDetectionBase=FACE_DETECTION_BASES[0], faceDetectionLoadPromise=null, tensorFlowLoadPromise=null, multiPersonDetectionAvailable=false, multiPersonDetectionError='', detectedFaceBoxes=[], multiplePeopleDetected=false, singlePersonFrames=0, lastFaceDetectionAt=-PERSON_CHECK_INTERVAL_MS, facingMode='environment', rafId=null, processing=false, lastVideoTime=-1, lastFrameAt=0;
  let trackingDimensionIndex=Math.max(0,TRACKING_DIMENSIONS.length-2), slowTrackingFrames=0, fastTrackingFrames=0;
  let processEvery=1000/30, model=null, modelConfig=null, modelReady=false, modelLoadPromise=null;
  let segmentFrames=[], previousFeatures=null, signerVisible=false, inferenceBusy=false, lastInferenceAt=0;
  let recognitionGeneration=0;
  let averageInferenceMs=0, inferenceCount=0, inferenceEveryMs=100, predictionLabel='', predictionStreak=0, lastCommitAt=0;
  let segmentActive=false, segmentCommitted=false, lastMotionAt=0, speechTimer=null, sentence=[], recognizedSigns=[], sentenceRevision=0, speechRevision=0;
  let installPrompt=null, voices=[];

  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const COMMIT_COOLDOWN_MS=700;
  const visible=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
  const drawable=(p,minConfidence=0)=>visible(p)&&(Number.isFinite(p.visibility)?p.visibility:Number.isFinite(p.presence)?p.presence:1)>=minConfidence;
  function adaptTrackingResolution(elapsed){if(elapsed>65){slowTrackingFrames++;fastTrackingFrames=0;if(slowTrackingFrames>=3){trackingDimensionIndex=Math.max(0,trackingDimensionIndex-1);slowTrackingFrames=0}}else if(elapsed<42){fastTrackingFrames++;slowTrackingFrames=0;if(fastTrackingFrames>=120){trackingDimensionIndex=Math.min(TRACKING_DIMENSIONS.length-1,trackingDimensionIndex+1);fastTrackingFrames=0}}else{slowTrackingFrames=0;fastTrackingFrames=0}}
  function toastMessage(text){toast.textContent=text;toast.classList.add('show');clearTimeout(toastMessage.timer);toastMessage.timer=setTimeout(()=>toast.classList.remove('show'),2800)}
  function setState(text,cls=''){state.textContent=text;state.className=('recognition-state '+cls).trim()}
  function cameraStatus(text,online=false){status.classList.toggle('online',online);status.innerHTML='<span></span> '+text}
  function updateSentence(){rawOutput.textContent=recognizedSigns.length?recognizedSigns.join(' | '):'Wala pang na-decode na senyas.';sentenceOutput.textContent=sentence.length?sentence.join(' '):'Ang na-decode na pangungusap ay lalabas dito.'}
  function stopSpeech(){speechRevision++;if(speechTimer)clearTimeout(speechTimer);speechTimer=null;if('speechSynthesis'in window)window.speechSynthesis.cancel()}
  function chooseFilipinoVoice(){if(!('speechSynthesis'in window))return null;voices=window.speechSynthesis.getVoices()||[];return voices.find(v=>/^fil(-|$)/i.test(v.lang))||voices.find(v=>/^tl(-|$)/i.test(v.lang))||voices.find(v=>/^en-PH$/i.test(v.lang))||null}
  function speakSentence(manual=false){if((!manual&&!autoSpeak?.checked)||!sentence.length||!('speechSynthesis'in window))return;const spokenRevision=sentenceRevision,speakingRevision=++speechRevision,utterance=new SpeechSynthesisUtterance(sentence.join(' '));utterance.lang='fil-PH';utterance.rate=.92;const voice=chooseFilipinoVoice();if(voice)utterance.voice=voice;utterance.onend=()=>{if(speakingRevision!==speechRevision||spokenRevision!==sentenceRevision)return;sentence=[];sentenceRevision++;updateSentence()};window.speechSynthesis.speak(utterance)}
  function scheduleSpeech(){if(speechTimer)clearTimeout(speechTimer);speechTimer=setTimeout(()=>{speechTimer=null;speakSentence()},1150)}
  function addWord(label,confidence){if(segmentCommitted||!Number.isFinite(confidence)||confidence<MIN_CONFIDENCE_PERCENT)return;const now=performance.now();if(now-lastCommitAt<COMMIT_COOLDOWN_MS)return;segmentCommitted=true;sentence.push(label);recognizedSigns.push(label);sentenceRevision++;lastCommitAt=now;updateSentence();toastMessage('Na-decode: '+label+' ('+Math.round(confidence)+'%)')}

  function showNoSign(message){detected.textContent='NO SIGN';confidenceValue.textContent='—';confidenceBar.style.width='0%';translation.textContent=modelReady?message:'Landmark tracking works; sign recognition is disabled until a validated FSL model is installed.';typePill.textContent=modelReady?'Waiting for sign':'Model not installed';setState(modelReady?'NO SIGN':'MODEL NOT READY');overlay.classList.remove('active')}
  function showMultiPersonStatus(){detected.textContent='MULTIPLE PEOPLE';confidenceValue.textContent='—';confidenceBar.style.width='0%';translation.textContent='Multiple faces detected. Holistic tracks one person at a time, so sign predictions are paused. Keep one signer in view.';typePill.textContent='Single-signer tracking';setState('MULTIPLE PEOPLE','verifying');overlay.classList.remove('active')}
  function showMultiPersonUnavailable(){detected.textContent='MULTI-PERSON CHECK OFF';confidenceValue.textContent='—';confidenceBar.style.width='0%';translation.textContent='Unable to check for additional people'+(multiPersonDetectionError?': '+multiPersonDetectionError:'')+'. Predictions are paused for safety.';typePill.textContent='Tracking check unavailable';setState('TRACKING WARNING','verifying');overlay.classList.remove('active')}
  function resetRecognitionSegment(){recognitionGeneration++;segmentFrames=[];previousFeatures=null;predictionLabel='';predictionStreak=0;segmentActive=false;segmentCommitted=false;lastProcessedAtForSequence=0;lastInferenceAt=0}
  function resizeCanvases(){const sourceW=video.videoWidth||video.clientWidth||1280,sourceH=video.videoHeight||video.clientHeight||720,s=Math.min(1,MAX_TRACKING_DIMENSION/Math.max(sourceW,sourceH)),w=Math.max(1,Math.round(sourceW*s)),h=Math.max(1,Math.round(sourceH*s));for(const c of [handCanvas,poseCanvas])if(c&&(c.width!==w||c.height!==h)){c.width=w;c.height=h}}
  function mirrorPreview(){const mirror=facingMode==='user';video.classList.toggle('mirror',mirror);handCanvas.classList.toggle('mirror',mirror);poseCanvas?.classList.toggle('mirror',mirror)}
  function getCanvasTransform(w,h){const rect=cameraStage.getBoundingClientRect(),sourceW=video.videoWidth||1280,sourceH=video.videoHeight||720;if(!rect.width||!rect.height)return{direct:true,width:w,height:h};const scale=Math.max(rect.width/sourceW,rect.height/sourceH);return{sourceW,sourceH,scale,offsetX:(rect.width-sourceW*scale)/2,offsetY:(rect.height-sourceH*scale)/2,scaleX:w/rect.width,scaleY:h/rect.height}}
  function canvasX(p,t){return t.direct?p.x*t.width:(p.x*t.sourceW*t.scale+t.offsetX)*t.scaleX}
  function canvasY(p,t){return t.direct?p.y*t.height:(p.y*t.sourceH*t.scale+t.offsetY)*t.scaleY}
  function drawLinks(context,points,links,w,color,lineScale,transform,minConfidence=0){if(!context||!points?.length)return;context.save();context.lineCap='round';context.lineJoin='round';const lineWidth=Math.max(1.5,w/600*lineScale);context.beginPath();for(const [a,b] of links){if(!drawable(points[a],minConfidence)||!drawable(points[b],minConfidence))continue;context.moveTo(canvasX(points[a],transform),canvasY(points[a],transform));context.lineTo(canvasX(points[b],transform),canvasY(points[b],transform))}context.strokeStyle='rgba(15,23,42,.9)';context.lineWidth=lineWidth+2;context.stroke();context.strokeStyle=color;context.lineWidth=lineWidth;context.stroke();context.restore()}
  function drawPoints(context,points,w,color,indices,transform,minConfidence=0){if(!context||!points?.length)return;const radius=Math.max(1.6,w/850),count=indices?indices.length:points.length;context.save();context.fillStyle='rgba(15,23,42,.9)';context.beginPath();for(let n=0;n<count;n++){const p=points[indices?indices[n]:n];if(!drawable(p,minConfidence))continue;const x=canvasX(p,transform),y=canvasY(p,transform);context.moveTo(x+radius+1,y);context.arc(x,y,radius+1,0,Math.PI*2)}context.fill();context.fillStyle=color;context.beginPath();for(let n=0;n<count;n++){const p=points[indices?indices[n]:n];if(!drawable(p,minConfidence))continue;const x=canvasX(p,transform),y=canvasY(p,transform);context.moveTo(x+radius,y);context.arc(x,y,radius,0,Math.PI*2)}context.fill();context.restore()}
  function drawPersonBoxes(context,boxes,w,transform){if(!context||boxes.length<2)return;context.save();const fontSize=Math.max(10,w/70);context.lineWidth=Math.max(2,w/420);context.font='700 '+fontSize+'px sans-serif';for(let i=0;i<boxes.length;i++){const box=boxes[i],topLeft={x:box.xCenter-box.width/2,y:box.yCenter-box.height/2},bottomRight={x:box.xCenter+box.width/2,y:box.yCenter+box.height/2},x=canvasX(topLeft,transform),y=canvasY(topLeft,transform),right=canvasX(bottomRight,transform),bottom=canvasY(bottomRight,transform);context.strokeStyle='#fb7185';context.strokeRect(x,y,right-x,bottom-y);context.fillStyle='rgba(136,19,55,.9)';context.fillRect(x,y-fontSize-6,context.measureText('FACE '+(i+1)).width+10,fontSize+6);context.fillStyle='#fff';context.fillText('FACE '+(i+1),x+5,y-4)}context.restore()}
  function drawResults(r){resizeCanvases();const w=handCanvas.width,h=handCanvas.height,transform=getCanvasTransform(w,h);handCtx.clearRect(0,0,w,h);if(poseCtx)poseCtx.clearRect(0,0,w,h);const pose=r.poseLandmarks||[],left=r.leftHandLandmarks||[],right=r.rightHandLandmarks||[],face=r.faceLandmarks||[];
    drawLinks(handCtx,pose,POSE_LINKS,w,'rgba(56,189,248,.95)',2.4,transform,.35);drawPoints(handCtx,pose,w,'#bae6fd',POSE_IDS,transform,.35);
    drawLinks(handCtx,left,HAND_LINKS,w,'#a78bfa',2.7,transform);drawPoints(handCtx,left,w,'#ddd6fe',null,transform);
    drawLinks(handCtx,right,HAND_LINKS,w,'#34d399',2.7,transform);drawPoints(handCtx,right,w,'#a7f3d0',null,transform);
    drawLinks(poseCtx||handCtx,face,FACE_OVAL_LINKS,w,'rgba(251,191,36,.7)',1.5,transform);drawLinks(poseCtx||handCtx,face,FACE_LINKS,w,'#fbbf24',1.5,transform);drawPoints(poseCtx||handCtx,face,w,'#fef3c7',FACE_IDS,transform);
    if(multiplePeopleDetected)drawPersonBoxes(poseCtx||handCtx,detectedFaceBoxes,w,transform);
    const bodyTracked=POSE_IDS.some(i=>drawable(pose[i],.35)),faceTracked=FACE_IDS.some(i=>drawable(face[i])),hands=Number(left.some(visible))+Number(right.some(visible)),tracked=[bodyTracked?'BODY':null,faceTracked?'FACE':null,hands?hands+' HAND'+(hands===1?'':'S'):null].filter(Boolean),trackingText=multiplePeopleDetected?'MULTI-PERSON · '+detectedFaceBoxes.length+' FACES':modelReady&&!multiPersonDetectionAvailable?'MULTI-CHECK OFF':tracked.length?tracked.join(' + '):'TRACKING: SEARCHING';if(trackingBadge.textContent!==trackingText)trackingBadge.textContent=trackingText;trackingBadge.classList.toggle('detected',tracked.length>0||multiplePeopleDetected);guide?.classList.toggle('is-hidden',tracked.length>0||multiplePeopleDetected);
  }

  function makeFeatures(r){return window.SignARFeatureExtractor?.extract(r,video.videoWidth||1280,video.videoHeight||720)||null}
  function measureMotion(v){if(!v||!previousFeatures)return 0;let sum=0;for(let i=0;i<FEATURE_DIMENSION;i++)sum+=Math.abs(v[i]-previousFeatures[i]);return sum/FEATURE_DIMENSION}

  async function warmupModel(loadedModel){
    const input=window.tf.zeros([1,SEQUENCE_LENGTH,FEATURE_DIMENSION]);
    let output=null;
    try{
      output=loadedModel.predict(input);
      const tensors=Array.isArray(output)?output:[output];
      await Promise.all(tensors.map(tensor=>tensor.data()));
    }finally{
      input.dispose();
      if(output)(Array.isArray(output)?output:[output]).forEach(tensor=>tensor.dispose?.());
    }
  }

  async function loadTensorFlowLibrary(){if(window.tf)return;if(!tensorFlowLoadPromise)tensorFlowLoadPromise=(async()=>{const failures=[];for(const src of TENSORFLOW_BASES){try{await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=src;script.crossOrigin='anonymous';script.onload=()=>window.tf?resolve():reject(new Error('TensorFlow.js loaded without defining tf.'));script.onerror=()=>reject(new Error('The script request failed.'));document.head.appendChild(script)});return}catch(error){failures.push(src+': '+error.message)}}throw new Error('Unable to load TensorFlow.js. Check your internet connection or content blocker. '+failures.join(' '))})();try{await tensorFlowLoadPromise}catch(error){tensorFlowLoadPromise=null;throw error}}

  async function loadModel(){if(modelLoadPromise)return modelLoadPromise;modelLoadPromise=(async()=>{
    try{
      const response=await fetch('./model/vocabulary.json',{cache:'no-store'});
      if(!response.ok)throw new Error('Vocabulary manifest is not present');
      modelConfig=await response.json();
      if(modelConfig.featureVersion!=='signar-fsl-456-v2')throw new Error('Feature schema mismatch; expected signar-fsl-456-v2');
      if(modelConfig.sequenceLength!==SEQUENCE_LENGTH||modelConfig.featureDimension!==FEATURE_DIMENSION)throw new Error('Vocabulary input-shape contract mismatch');
      if(!modelConfig.trained||!modelConfig.modelPath){modelReady=false;modelStatus.textContent='No validated multi-sign model is installed. Landmark tracking works, but sign recognition is disabled until a reviewed model is trained and installed.';return}
      if(!Array.isArray(modelConfig.labels)||modelConfig.labels.length<2||modelConfig.labels.some(label=>typeof label!=='string'||!label.trim()))throw new Error('A deployable model must contain at least two sign labels');
      if(modelConfig.outputClasses!==undefined&&modelConfig.outputClasses!==modelConfig.labels.length)throw new Error('Manifest output class count does not match its sign labels');
      await loadTensorFlowLibrary();await window.tf.ready();
      const modelUrl=new URL(modelConfig.modelPath,window.location.href).href;
      model=await window.tf.loadLayersModel(modelUrl);
      try{
        const input=model.inputs?.[0]?.shape||[],output=model.outputs?.[0]?.shape||[];
        if(input.length!==3||input[1]!==SEQUENCE_LENGTH||input[2]!==FEATURE_DIMENSION)throw new Error('Model input shape mismatch; expected [batch,30,456]');
        if(output.at(-1)!==modelConfig.labels.length)throw new Error('Model class count mismatch');
        await warmupModel(model);
      }catch(error){model.dispose();model=null;throw error}
      modelReady=true;
      modelStatus.textContent='Validated multi-sign model loaded. Prediction labels are read from its manifest.';
    }catch(error){modelReady=false;console.info('FSL model unavailable:',error.message);modelStatus.textContent='Model unavailable: '+error.message+' Sign recognition is disabled; landmark tracking remains available.'}
  })();return modelLoadPromise}

  async function infer(){if(!modelReady||!model||multiplePeopleDetected||!multiPersonDetectionAvailable||!segmentActive||segmentCommitted||inferenceBusy||segmentFrames.length<MIN_SEGMENT_FRAMES||performance.now()-lastInferenceAt<inferenceEveryMs)return;inferenceBusy=true;lastInferenceAt=performance.now();const generation=recognitionGeneration,started=performance.now();let input=null,output=null;
    try{const sampled=window.SignARFeatureExtractor.resampleSequence(segmentFrames,SEQUENCE_LENGTH);
      input=window.tf.tensor3d([sampled],[1,SEQUENCE_LENGTH,FEATURE_DIMENSION]);output=model.predict(input);const tensor=Array.isArray(output)?output[0]:output;const p=await tensor.data();if(generation!==recognitionGeneration||multiplePeopleDetected||!multiPersonDetectionAvailable)return;let best=0;for(let i=1;i<p.length;i++)if(p[i]>p[best])best=i;
      const label=modelConfig.labels?.[best]||best,confidence=clamp(p[best]*100,0,100);if(!label)return;
      if(!Number.isFinite(confidence)||confidence<MIN_CONFIDENCE_PERCENT){
        predictionLabel='';predictionStreak=0;detected.textContent='LOW CONFIDENCE';confidenceValue.textContent=Number.isFinite(confidence)?confidence.toFixed(1)+'%':'—';confidenceBar.style.width=Number.isFinite(confidence)?confidence+'%':'0%';
        translation.textContent='Below the 99% model-score threshold. No sign was decoded.';
        typePill.textContent='Waiting for 99% score';setState('LOW CONFIDENCE');overlay.classList.remove('active');return
      }
      if(predictionLabel===label)predictionStreak++;else{predictionLabel=label;predictionStreak=1}
      detected.textContent=label;confidenceValue.textContent=Math.round(confidence)+'%';confidenceBar.style.width=Math.round(confidence)+'%';
      translation.textContent='Prediction ng temporal model: '+label;
      typePill.textContent='TensorFlow.js sequence model';arSign.textContent=label;arMeaning.textContent=label;overlay.classList.add('active');setState('PREDICTING','verifying');
      // Require a high model score and temporal stability before adding a sign.
      if(segmentActive&&!segmentCommitted&&predictionStreak>=2&&signerVisible)addWord(label,confidence);
    }catch(error){console.error('Inference error:',error);translation.textContent='Nagkaroon ng error sa model inference.';setState('MODEL ERROR')}
    finally{input?.dispose();if(output)(Array.isArray(output)?output:[output]).forEach(t=>t.dispose?.());const elapsed=performance.now()-started;inferenceCount++;averageInferenceMs+=(elapsed-averageInferenceMs)/inferenceCount;const inferenceDelay=averageInferenceMs>85?180:averageInferenceMs>50?130:100;inferenceEveryMs=inferenceDelay;inferenceBusy=false}
  }

  function onFaceDetectionResults(r){detectedFaceBoxes=(r.detections||[]).map(d=>d.boundingBox).filter(box=>box&&Number.isFinite(box.xCenter)&&Number.isFinite(box.yCenter)&&Number.isFinite(box.width)&&Number.isFinite(box.height)&&box.width>0&&box.height>0);if(detectedFaceBoxes.length>=2){if(!multiplePeopleDetected)resetRecognitionSegment();multiplePeopleDetected=true;singlePersonFrames=0}else{singlePersonFrames++;if(singlePersonFrames>=3&&multiplePeopleDetected){multiplePeopleDetected=false;resetRecognitionSegment()}}}
  async function checkFaceDetection(image,time){if(!faceDetection||time-lastFaceDetectionAt<PERSON_CHECK_INTERVAL_MS)return;try{await faceDetection.send({image})}catch(error){console.error('Multi-person detection failed:',error);faceDetection=null;multiPersonDetectionAvailable=false;multiPersonDetectionError=error.message;multiplePeopleDetected=false;detectedFaceBoxes=[];resetRecognitionSegment()}finally{lastFaceDetectionAt=performance.now()}}
  function onResults(r){drawResults(r);if(!modelReady){showNoSign('Landmark tracking works; a validated model is required for sign recognition.');return}if(multiplePeopleDetected){showMultiPersonStatus();return}if(!multiPersonDetectionAvailable){showMultiPersonUnavailable();return}const vector=makeFeatures(r);signerVisible=!!vector;guide?.classList.toggle('is-hidden',signerVisible);
    if(!vector){resetRecognitionSegment();signerVisible=false;showNoSign('Walang nakikitang katawan sa tracking area.');return}
    const now=performance.now(),energy=measureMotion(vector);
    if(energy>=.0055){lastMotionAt=now;if(!segmentActive){segmentActive=true;segmentCommitted=false;segmentFrames=[];predictionLabel='';predictionStreak=0;lastProcessedAtForSequence=0;lastInferenceAt=0}stopSpeech()}
    previousFeatures=vector;
    if(segmentActive&&now-lastProcessedAtForSequence>=SEQUENCE_SAMPLE_MS){segmentFrames.push(vector);if(segmentFrames.length>MAX_SEGMENT_FRAMES)segmentFrames.shift();lastProcessedAtForSequence=now;void infer()}
    if(segmentActive&&segmentFrames.length>=MIN_SEGMENT_FRAMES&&now-lastMotionAt>QUIET_TO_END_MS&&(segmentCommitted||now-lastMotionAt>1400)&&!inferenceBusy){const hadPrediction=segmentCommitted;segmentActive=false;segmentCommitted=false;predictionLabel='';predictionStreak=0;segmentFrames=[];if(hadPrediction&&sentence.length)scheduleSpeech();showNoSign('Tapos na ang senyas.')}
  }
  let lastProcessedAtForSequence=0;

  async function loadHolisticLibrary(){if(typeof window.Holistic==='function')return;if(!holisticLoadPromise)holisticLoadPromise=(async()=>{const failures=[];
      for(const base of HOLISTIC_BASES){try{await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=base+'/holistic.js';script.crossOrigin='anonymous';script.onload=()=>typeof window.Holistic==='function'?resolve():reject(new Error('The script loaded without defining Holistic.'));script.onerror=()=>reject(new Error('The script request failed.'));document.head.appendChild(script)});holisticBase=base;return}catch(error){failures.push(base+': '+error.message)}}
      throw new Error('Unable to load MediaPipe Holistic. Check your internet connection or content blocker and retry. '+failures.join(' '));
    })();try{await holisticLoadPromise}catch(error){holisticLoadPromise=null;throw error}}
  async function loadFaceDetectionLibrary(){if(typeof window.FaceDetection==='function')return;if(!faceDetectionLoadPromise)faceDetectionLoadPromise=(async()=>{const failures=[];for(const base of FACE_DETECTION_BASES){try{await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=base+'/face_detection.js';script.crossOrigin='anonymous';script.onload=()=>typeof window.FaceDetection==='function'?resolve():reject(new Error('The script loaded without defining FaceDetection.'));script.onerror=()=>reject(new Error('The script request failed.'));document.head.appendChild(script)});faceDetectionBase=base;return}catch(error){failures.push(base+': '+error.message)}}throw new Error('Unable to load MediaPipe Face Detection. Check your internet connection or content blocker. '+failures.join(' '))})();try{await faceDetectionLoadPromise}catch(error){faceDetectionLoadPromise=null;throw error}}
  async function initHolistic(){if(holistic)return;await loadHolisticLibrary();
    holistic=new window.Holistic({locateFile:file=>holisticBase+'/'+file});holistic.setOptions({modelComplexity:1,smoothLandmarks:true,refineFaceLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});holistic.onResults(onResults);
    if(typeof holistic.initialize==='function')await holistic.initialize();
    await loadModel()
    if(!modelReady)return;
    try{await loadFaceDetectionLibrary();faceDetection=new window.FaceDetection({locateFile:file=>faceDetectionBase+'/'+file});faceDetection.setOptions({model:'full',minDetectionConfidence:.6});faceDetection.onResults(onFaceDetectionResults);if(typeof faceDetection.initialize==='function')await faceDetection.initialize();multiPersonDetectionAvailable=true}catch(error){faceDetection=null;multiPersonDetectionAvailable=false;multiPersonDetectionError=error.message;console.error('Multi-person detection unavailable:',error)}
  }
  async function openCamera(){const ideal={width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:30}};
    try{return await navigator.mediaDevices.getUserMedia({video:{...ideal,facingMode:{exact:facingMode}},audio:false})}
    catch(error){if(['OverconstrainedError','NotFoundError','TypeError'].includes(error?.name)){try{return await navigator.mediaDevices.getUserMedia({video:ideal,audio:false})}catch{return await navigator.mediaDevices.getUserMedia({video:true,audio:false})}}throw error}
  }
  function cameraError(e){if(!window.isSecureContext)return 'Camera access requires HTTPS or localhost.';if(e?.name==='NotAllowedError')return 'Pinayagan ang camera access? Payagan ang camera sa browser settings.';if(e?.name==='NotFoundError'||e?.name==='OverconstrainedError')return 'Walang compatible na camera sa device.';if(e?.name==='NotReadableError')return 'Maaaring ginagamit ng ibang app ang camera.';return 'Camera error: '+(e?.message||e?.name||'unknown')}
  async function startCamera(){try{if(!window.isSecureContext)throw new Error('HTTPS required');if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera API unavailable');await stopCamera(false);cameraStatus('Loading tracking models…');await initHolistic();stream=await openCamera();video.srcObject=stream;await video.play();
      multiplePeopleDetected=false;singlePersonFrames=0;detectedFaceBoxes=[];lastFaceDetectionAt=-PERSON_CHECK_INTERVAL_MS;
      const actual=stream.getVideoTracks()[0]?.getSettings?.().facingMode;if(actual)facingMode=actual;mirrorPreview();resizeCanvases();video.classList.add('active');placeholder.style.display='none';liveBadge.classList.add('show');trackingBadge.classList.add('show');startBtn.disabled=true;stopBtn.disabled=false;switchBtn.textContent=facingMode==='environment'?'Use front camera':'Use back camera';cameraStatus('Camera on',true);lastVideoTime=-1;lastFrameAt=0;rafId=requestAnimationFrame(frameLoop);
      toastMessage(modelReady?'Camera ready. Trained decoder loaded.':'Camera ready. Landmark tracking works; a validated model is required for sign predictions.');
    }catch(e){console.error(e);await stopCamera(false);cameraStatus('Camera unavailable');translation.textContent=cameraError(e);toastMessage(cameraError(e))}}
  async function frameLoop(time){if(!stream)return;if(!processing&&video.readyState>=2&&video.currentTime!==lastVideoTime&&time-lastFrameAt>=processEvery){processing=true;lastFrameAt=time;lastVideoTime=video.currentTime;const started=performance.now();try{const sourceW=video.videoWidth||1280,sourceH=video.videoHeight||720,scale=Math.min(1,TRACKING_DIMENSIONS[trackingDimensionIndex]/Math.max(sourceW,sourceH)),width=Math.max(1,Math.round(sourceW*scale)),height=Math.max(1,Math.round(sourceH*scale));if(processCanvas.width!==width)processCanvas.width=width;if(processCanvas.height!==height)processCanvas.height=height;processCtx.drawImage(video,0,0,width,height);
        if(faceDetection&&time-lastFaceDetectionAt>=PERSON_CHECK_INTERVAL_MS){const faceScale=Math.min(1,PERSON_CHECK_DIMENSION/Math.max(sourceW,sourceH)),faceWidth=Math.max(1,Math.round(sourceW*faceScale)),faceHeight=Math.max(1,Math.round(sourceH*faceScale));if(personCheckCanvas.width!==faceWidth)personCheckCanvas.width=faceWidth;if(personCheckCanvas.height!==faceHeight)personCheckCanvas.height=faceHeight;personCheckCtx.drawImage(video,0,0,faceWidth,faceHeight);await checkFaceDetection(personCheckCanvas,time)}
        await holistic.send({image:processCanvas})}catch(e){console.error(e);setState('TRACKING ERROR')}finally{const ms=performance.now()-started;adaptTrackingResolution(ms);processEvery=ms>75?1000/12:ms>48?1000/18:1000/30;processing=false}}rafId=requestAnimationFrame(frameLoop)}
  async function stopCamera(show=true){if(rafId)cancelAnimationFrame(rafId);rafId=null;processing=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.classList.remove('active');placeholder.style.display='flex';liveBadge.classList.remove('show');trackingBadge.classList.remove('show','detected');handCtx.clearRect(0,0,handCanvas.width,handCanvas.height);if(poseCtx)poseCtx.clearRect(0,0,poseCanvas.width,poseCanvas.height);startBtn.disabled=false;stopBtn.disabled=true;segmentFrames=[];previousFeatures=null;signerVisible=false;segmentActive=false;segmentCommitted=false;guide?.classList.remove('is-hidden');if(show){cameraStatus('Camera off');showNoSign('Walang kasalukuyang senyas.')}}
  async function switchCamera(){const desired=facingMode==='environment'?'user':'environment';facingMode=desired;if(stream){await stopCamera(false);await startCamera()}else{mirrorPreview();switchBtn.textContent=facingMode==='environment'?'Use front camera':'Use back camera'}}
  navButtons.forEach(btn=>btn.addEventListener('click',()=>{navButtons.forEach(b=>b.classList.toggle('active',b===btn));sections.forEach(s=>s.classList.toggle('active-section',s.id===btn.dataset.target))}));
  startBtn.addEventListener('click',startCamera);stopBtn.addEventListener('click',()=>stopCamera(true));switchBtn.addEventListener('click',switchCamera);
  fullscreenBtn.addEventListener('click',async()=>{try{if(!document.fullscreenElement)await cameraStage.requestFullscreen();else await document.exitFullscreen()}catch{toastMessage('Fullscreen is not available in this browser.')}});
  deleteBtn.addEventListener('click',()=>{sentence.pop();recognizedSigns.pop();sentenceRevision++;updateSentence();stopSpeech()});clearBtn.addEventListener('click',()=>{sentence=[];recognizedSigns=[];sentenceRevision++;updateSentence();stopSpeech()});speakBtn.addEventListener('click',()=>{stopSpeech();speakSentence(true)});
  window.addEventListener('resize',resizeCanvases);window.addEventListener('orientationchange',()=>setTimeout(resizeCanvases,250));window.addEventListener('beforeunload',()=>{void stopCamera(false);stopSpeech()});
  window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();installPrompt=e;installBtn.hidden=false});installBtn.addEventListener('click',async()=>{if(!installPrompt)return;installPrompt.prompt();await installPrompt.userChoice;installPrompt=null;installBtn.hidden=true});
  if('speechSynthesis'in window)window.speechSynthesis.onvoiceschanged=chooseFilipinoVoice;
  if('serviceWorker'in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(console.warn));
  switchBtn.textContent='Use front camera';mirrorPreview();updateSentence();showNoSign('Landmark tracking is ready. A validated FSL model is required for sign predictions.');
})();