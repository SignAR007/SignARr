/* Shared SignAR FSL feature extraction. Keep this schema identical for collection, training, and inference. */
(() => {
  'use strict';
  const DIMENSION = 456;
  const MISSING = -3.0;
  const MISSING_THRESHOLD = -2.75;
  const COORD_LIMIT = 2.5;
  const FACE_IDS = [10,152,234,454,1,4,33,133,159,145,362,263,386,374,61,291,13,14,70,300,105,334,78,308];
  const BODY_ANCHOR_IDS = [0,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pointOK = p => !!p && Number.isFinite(p.x) && Number.isFinite(p.y)
    && p.x >= -0.5 && p.x <= 1.5 && p.y >= -0.5 && p.y <= 1.5;
  const confidence = p => !pointOK(p) ? 0 : clamp(Number.isFinite(p.visibility) ? p.visibility : 1, 0, 1);
  const poseOK = (p, threshold = 0.15) => pointOK(p) && confidence(p) >= threshold;
  const pointList = values => (values || []).filter(pointOK);

  function toPixels(p, width, height) {
    return { x: p.x * width, y: p.y * height, z: (Number.isFinite(p.z) ? p.z : 0) * width };
  }
  function distancePx(a, b, width, height) {
    const pa = toPixels(a, width, height), pb = toPixels(b, width, height);
    return Math.hypot(pa.x - pb.x, pa.y - pb.y);
  }
  function bounds(points, width, height) {
    if (!points.length) return null;
    const pixels = points.map(p => toPixels(p, width, height));
    const xs = pixels.map(p => p.x), ys = pixels.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    return { minX, maxX, minY, maxY, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2,
      width: maxX - minX, height: maxY - minY, diagonal: Math.hypot(maxX - minX, maxY - minY) };
  }
  function extract(results, width = 1280, height = 720) {
    const pose = results?.poseLandmarks || [];
    const face = results?.faceLandmarks || [];
    const hands = [results?.leftHandLandmarks || [], results?.rightHandLandmarks || []];
    const minDimension = Math.max(1, Math.min(width, height));
    const reliablePose = BODY_ANCHOR_IDS.map(i => pose[i]).filter(p => poseOK(p));
    const poseBounds = bounds(reliablePose, width, height);
    const poseAvailable = reliablePose.length >= 1 && !!poseBounds;

    const facePoints = FACE_IDS.map(i => face[i]).filter(pointOK);
    const faceBounds = bounds(facePoints, width, height);
    const faceAvailable = facePoints.length >= 1 && !!faceBounds;

    const handInfo = hands.map(hand => {
      const valid = pointList(hand).slice(0, 21);
      const box = bounds(valid, width, height);
      return { points: hand || [], validCount: valid.length, box,
        available: valid.length >= 1 && !!box };
    });
    const handsAvailable = handInfo.some(h => h.available);

    const shoulderPair = poseOK(pose[11], 0.25) && poseOK(pose[12], 0.25);
    const hipsReliable = poseOK(pose[23], 0.25) && poseOK(pose[24], 0.25);
    let anchorX, anchorY, scale;

    // Prefer a torso anchor when shoulders exist. Hips are optional.
    if (shoulderPair) {
      const leftShoulder = toPixels(pose[11], width, height);
      const rightShoulder = toPixels(pose[12], width, height);
      const shoulderWidth = Math.max(1, Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y));
      const shoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
      if (hipsReliable) {
        const leftHip = toPixels(pose[23], width, height), rightHip = toPixels(pose[24], width, height);
        const hip = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
        anchorX = (shoulder.x + hip.x) / 2; anchorY = (shoulder.y + hip.y) / 2;
        scale = Math.max(1, Math.hypot(hip.x - shoulder.x, hip.y - shoulder.y));
      } else {
        anchorX = shoulder.x; anchorY = shoulder.y;
        scale = Math.max(1, shoulderWidth * 1.35);
      }
    } else if (faceAvailable) {
      // If the torso is cropped, a face anchor still gives the hands consistent face-relative positions.
      anchorX = faceBounds.centerX; anchorY = faceBounds.centerY;
      const eyeA = face[33], eyeB = face[263];
      const eyeDistance = pointOK(eyeA) && pointOK(eyeB) ? distancePx(eyeA, eyeB, width, height) : 0;
      scale = Math.max(1, eyeDistance, faceBounds.diagonal * 0.45);
    } else if (poseAvailable) {
      // Partial pose: normalize around the visible upper-body landmarks instead of rejecting the frame.
      anchorX = poseBounds.centerX; anchorY = poseBounds.centerY;
      scale = Math.max(1, poseBounds.diagonal * 0.8, minDimension * 0.03);
    } else if (handsAvailable) {
      const validHandPoints = handInfo.filter(h => h.available).flatMap(h => pointList(h.points).slice(0, 21));
      const handBounds = bounds(validHandPoints, width, height);
      // With no body/face anchor, keep the frame center as origin so directional hand movement is not normalized away.
      anchorX = width / 2; anchorY = height / 2;
      const palmWidths = handInfo.filter(h => h.available).map(h => {
        const p = h.points;
        return pointOK(p[5]) && pointOK(p[17]) ? distancePx(p[5], p[17], width, height) : 0;
      }).filter(v => v > 0);
      const averagePalm = palmWidths.length ? palmWidths.reduce((a, b) => a + b, 0) / palmWidths.length : 0;
      scale = Math.max(1, averagePalm * 2.5, handBounds.diagonal * 0.8, minDimension * 0.025);
    } else {
      // Avoid saving empty/background frames when no sign-relevant landmarks were detected.
      return null;
    }

    const norm = p => {
      if (!pointOK(p)) return [0, 0, 0];
      const px = toPixels(p, width, height);
      return [clamp((px.x - anchorX) / scale, -COORD_LIMIT, COORD_LIMIT),
        clamp((px.y - anchorY) / scale, -COORD_LIMIT, COORD_LIMIT),
        clamp(px.z / scale, -COORD_LIMIT, COORD_LIMIT)];
    };
    const vector = [];

    // Pose is represented by a zero visibility channel for missing/low-confidence joints.
    for (let i = 0; i < 33; i++) {
      const p = pose[i], conf = confidence(p);
      if (poseOK(p, 0.05)) vector.push(...norm(p), conf);
      else vector.push(0, 0, 0, 0);
    }

    // A whole hand may be absent, and individual landmarks may be cropped. -3 is a reserved missing-data marker.
    for (const info of handInfo) {
      const hand = info.points;
      if (!info.available) {
        for (let i = 0; i < 21; i++) vector.push(MISSING, MISSING, MISSING, MISSING, MISSING, MISSING);
        continue;
      }
      const wrist = pointOK(hand[0]) ? hand[0] : null;
      const palm = pointOK(hand[5]) && pointOK(hand[17])
        ? Math.max(1, distancePx(hand[5], hand[17], width, height))
        : Math.max(1, (info.box?.diagonal || minDimension * 0.05) * 0.5);
      for (let i = 0; i < 21; i++) {
        const p = hand[i];
        if (!pointOK(p)) {
          vector.push(MISSING, MISSING, MISSING, MISSING, MISSING, MISSING);
          continue;
        }
        const global = norm(p);
        if (wrist) {
          const local = toPixels(p, width, height), wristPx = toPixels(wrist, width, height);
          vector.push(...global,
            clamp((local.x - wristPx.x) / palm, -COORD_LIMIT, COORD_LIMIT),
            clamp((local.y - wristPx.y) / palm, -COORD_LIMIT, COORD_LIMIT),
            clamp((local.z - wristPx.z) / palm, -COORD_LIMIT, COORD_LIMIT));
        } else {
          vector.push(...global, MISSING, MISSING, MISSING);
        }
      }
    }

    for (const i of FACE_IDS) {
      if (pointOK(face[i])) vector.push(...norm(face[i]));
      else vector.push(MISSING, MISSING, MISSING);
    }
    if (vector.length !== DIMENSION) throw new Error(`Feature schema error: expected ${DIMENSION}, got ${vector.length}`);
    return vector;
  }

  function resampleSequence(frames, target = 30) {
    if (!Array.isArray(frames) || frames.length < 2 || !Number.isInteger(target) || target < 2)
      throw new Error('A sequence needs at least two frames and a target length >= 2');
    if (frames.some(frame => !Array.isArray(frame) || frame.length !== DIMENSION || frame.some(value => !Number.isFinite(value))))
      throw new Error('Invalid landmark feature frame');
    const last = frames.length - 1, out = [];
    for (let i = 0; i < target; i++) {
      const position = i * last / (target - 1), lo = Math.floor(position), hi = Math.min(lo + 1, last), weight = position - lo;
      const frame = new Array(DIMENSION);
      for (let j = 0; j < DIMENSION; j++) {
        const a = frames[lo][j], b = frames[hi][j];
        // Do not blend a missing landmark sentinel into plausible-looking coordinates.
        frame[j] = (a <= MISSING_THRESHOLD || b <= MISSING_THRESHOLD)
          ? (weight < 0.5 ? a : b)
          : a * (1 - weight) + b * weight;
      }
      out.push(frame);
    }
    return out;
  }
  window.SignARFeatureExtractor = Object.freeze({
    version: 'signar-fsl-456-v2', dimension: DIMENSION, samplingFps: 15, sequenceLength: 30,
    faceIds: FACE_IDS, missingValue: MISSING, missingThreshold: MISSING_THRESHOLD, extract, resampleSequence
  });
})();
