/* Node.js smoke tests for partial-landmark extraction; no third-party packages required. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
global.window = {};
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'shared', 'fsl-features.js'), 'utf8'));
const extractor = global.window.SignARFeatureExtractor;
const decoderWindow = {};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'shared', 'fsl-features.js'), 'utf8'), { window: decoderWindow });
const decoderExtractor = decoderWindow.SignARFeatureExtractor;
assert.equal(extractor.version, 'signar-fsl-456-v2');
assert.equal(extractor.dimension, 456);
assert.equal(decoderExtractor.version, extractor.version);
assert.equal(decoderExtractor.dimension, extractor.dimension);
const assertDecoderParity = (results, label) => {
  const collected = extractor.extract(results, 640, 480);
  const decoded = decoderExtractor.extract(results, 640, 480);
  assert.equal(JSON.stringify(decoded), JSON.stringify(collected), label + ' feature parity');
};
const point = (x, y, z = 0, visibility = 1) => ({ x, y, z, visibility });
const makeHand = (offsetX = 0.5, offsetY = 0.5) => Array.from({ length: 21 }, (_, i) => {
  const col = i % 5, row = Math.floor(i / 5);
  return { x: offsetX + (col - 2) * 0.012, y: offsetY + row * 0.012, z: 0 };
});
const makeFace = () => {
  const face = Array.from({ length: 468 }, (_, i) => ({ x: 0.5, y: 0.35, z: 0 }));
  for (const [n, id] of extractor.faceIds.entries()) face[id] = { x: 0.42 + (n % 6) * 0.025, y: 0.3 + Math.floor(n / 6) * 0.025, z: 0 };
  return face;
};
const assertFrame = (frame, label) => {
  assert.ok(Array.isArray(frame), label + ' should produce a frame');
  assert.equal(frame.length, 456, label + ' has fixed feature width');
  assert.ok(frame.every(Number.isFinite), label + ' produces finite values');
};

// One hand only: no full body, other hand, or face required.
const oneHand = extractor.extract({ leftHandLandmarks: makeHand(), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 640, 480);
assertFrame(oneHand, 'one-hand-only clip');
assert.ok(oneHand.slice(132, 258).some(v => v !== extractor.missingValue));
assert.ok(oneHand.slice(258, 384).every(v => v === extractor.missingValue));
assert.ok(oneHand.slice(384).every(v => v === extractor.missingValue));
assertDecoderParity({ leftHandLandmarks: makeHand(), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 'one-hand-only clip');

const oneHandMoved = extractor.extract({ leftHandLandmarks: makeHand(0.7, 0.5), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 640, 480);
assertFrame(oneHandMoved, 'moved one-hand-only clip');
assert.notDeepEqual(oneHand.slice(132, 258), oneHandMoved.slice(132, 258), 'hand-only directional movement should remain represented');
assertDecoderParity({ leftHandLandmarks: makeHand(0.7, 0.5), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 'moved one-hand-only clip');

// Face plus a hand and partial pose, with shoulders and hips absent.
const partialPose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
partialPose[13] = point(0.4, 0.45, 0, 0.95);
partialPose[15] = point(0.35, 0.5, 0, 0.95);
const faceAndPartialPose = extractor.extract({ poseLandmarks: partialPose, faceLandmarks: makeFace(), leftHandLandmarks: [], rightHandLandmarks: makeHand(0.35, 0.5) }, 640, 480);
assertFrame(faceAndPartialPose, 'partial upper-body + face + one hand');
assertDecoderParity({ poseLandmarks: partialPose, faceLandmarks: makeFace(), leftHandLandmarks: [], rightHandLandmarks: makeHand(0.35, 0.5) }, 'partial upper-body + face + one hand');

// Face-only footage is accepted when several selected facial landmarks are visible.
const faceOnly = extractor.extract({ faceLandmarks: makeFace(), poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 640, 480);
assertFrame(faceOnly, 'face-only clip');
assertDecoderParity({ faceLandmarks: makeFace(), poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'face-only clip');

// Partial upper-body pose is accepted without shoulders, hips, hands, or face.
const armOnlyPose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
armOnlyPose[15] = point(0.42, 0.40, 0, 0.9);
armOnlyPose[16] = point(0.58, 0.46, 0, 0.9);
const poseOnly = extractor.extract({ poseLandmarks: armOnlyPose, faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 640, 480);
assertFrame(poseOnly, 'partial-pose-only clip');
assertDecoderParity({ poseLandmarks: armOnlyPose, faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'partial-pose-only clip');

// Missing all meaningful regions is still rejected so blank/background video isn't saved.
assert.equal(extractor.extract({ poseLandmarks: [], faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }), null);
assertDecoderParity({ poseLandmarks: [], faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'empty frame');

// Missing-value sentinels must remain discrete during temporal resampling.
const a = new Array(456).fill(0), b = new Array(456).fill(0);
a[132] = extractor.missingValue;
const resampled = extractor.resampleSequence([a, b], 3);
assert.equal(JSON.stringify(decoderExtractor.resampleSequence([a, b], 3)), JSON.stringify(resampled), 'decoder and collector sequence resampling should match');
assert.equal(resampled[0][132], extractor.missingValue);
assert.equal(resampled[1][132], 0);
assert.equal(resampled[2][132], 0);
console.log('Partial-landmark feature tests passed.');
