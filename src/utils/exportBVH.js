import { OrientationEstimator } from './orientationEstimator.js';

// ── BVH Exporter ─────────────────────────────────────────────────────────────
// Converts MediaPipe pose landmark frames to a BVH animation file.
// Matches Mixamo/Blender BVH export convention exactly.
// Includes full finger hierarchy driven by MediaPipe HandLandmarker data,
// with Gemini Vision as a fallback when hands are not detected.

const SCALE = 100;

const avg = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (v, s) => [v[0] * s, v[1] * s, v[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (v) => Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
const norm = (v) => { const l = len(v) || 1e-8; return [v[0] / l, v[1] / l, v[2] / l]; };
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];

// Rodrigues rotation
function rotateAround(vec, ax, rad) {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return add(add(scale(vec, c), scale(cross(ax, vec), s)), scale(ax, dot(ax, vec) * (1 - c)));
}

// ── Map Gemini's wrist_rotation string to a BVH Y-rotation degree ─────────────
function wristRotationDeg(rotation, side) {
  const map = { supinated: 60, neutral: 0, pronated: -60 };
  const deg = map[rotation] ?? 0;
  return side === 'right' ? -deg : deg;
}

// ── Wrist flex from HandLandmarker palm direction ─────────────────────────────
function getWristFlex(frame, side, forearmDir, yawDeg = 0) {
  const handLms = frame.handData?.[side]?.landmarks;
  if (!handLms || handLms.length < 10) return 0;

  const wrist = handLms[0];
  const middleMCP = handLms[9];

  const wristPos = preRotateY([
    -wrist.x * SCALE,
    -wrist.y * SCALE,
    wrist.z * SCALE,
  ], yawDeg);

  const middlePos = preRotateY([
    -middleMCP.x * SCALE,
    -middleMCP.y * SCALE,
    middleMCP.z * SCALE,
  ], yawDeg);

  const palmDir = norm(sub(middlePos, wristPos));

  // Fix: Re-align projection reference to remove the 90-degree twist offset
  const cosA = Math.max(-1, Math.min(1, dot(norm(forearmDir), palmDir)));

  // Adjusted rest offset matching anatomical T-pose baselines exactly
  const WRIST_REST_OFFSET = 0; 

  const flexDeg = Math.acos(cosA) * (180 / Math.PI) - WRIST_REST_OFFSET;

  return Math.max(-70, Math.min(70, flexDeg));
}

// ── Get finger angles for a hand, preferring MediaPipe then Gemini ────────────
function resolveFingerAngles(frame, side) {
  if (frame.handData?.[side]?.fingerAngles) {
    return frame.handData[side].fingerAngles;
  }
  if (frame.geminiFingerPoses?.hands) {
    const hand = frame.geminiFingerPoses.hands.find(h => h.side === side && h.confidence > 0.4);
    if (hand?.fingers) return hand.fingers;
  }
  return null;
}

// ── Pre-rotation functions ────────────────────────────────────────────────────
function preRotateY([x, y, z], yawDeg) {
  const rad = -yawDeg * (Math.PI / 180);
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [x * c - z * s, y, x * s + z * c];
}

function preRotateGravity([x, y, z], gravityAngleDeg) {
  if (!gravityAngleDeg || Math.abs(gravityAngleDeg) < 5) return [x, y, z];
  const rad = -gravityAngleDeg * (Math.PI / 180);
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [x * c - y * s, x * s + y * c, z];
}

// ── Coordinate extraction with yaw and gravity pre-rotation ──────────────────
function mp(lms, worldLms, idx, yawDeg = 0, gravityAngleDeg = 0) {
  const src = worldLms?.[idx] ?? lms[idx];
  if (!src) return [0, 0, 0];
  if (worldLms?.[idx]) {
    const raw = [-src.x * SCALE, -src.y * SCALE, src.z * SCALE];
    const yawRotated = preRotateY(raw, yawDeg);
    return preRotateGravity(yawRotated, gravityAngleDeg);
  }
  return [-src.x * SCALE, -src.y * SCALE, 0];
}

// ── Normalize yaw to [-180, 180] ──────────────────────────────────────────────
function wrapYaw(yaw) {
  let w = yaw % 360;
  if (w > 180) w -= 360;
  if (w < -180) w += 360;
  return w;
}

// ── Bone length cache ─────────────────────────────────────────────────────────
let cachedLengths = null;

function resetBoneLengthCache() {
  cachedLengths = null;
}

function getBoneLengths(rawLKnee, rawRKnee, lAnkle, rAnkle, rawLElbow, rawRElbow, lWrist, rWrist, leftHip, rightHip, leftSho, rightSho) {
  if (cachedLengths) return cachedLengths;
  cachedLengths = {
    lThigh: len(sub(rawLKnee, leftHip)) || 1,
    rThigh: len(sub(rawRKnee, rightHip)) || 1,
    lShin: len(sub(lAnkle, rawLKnee)) || 1,
    rShin: len(sub(rAnkle, rawRKnee)) || 1,
    lUpper: len(sub(rawLElbow, leftSho)) || 1,
    rUpper: len(sub(rawRElbow, rightSho)) || 1,
    lFore: len(sub(lWrist, rawLElbow)) || 1,
    rFore: len(sub(rWrist, rawRElbow)) || 1,
  };
  return cachedLengths;
}

// ── Foot floor clamp ──────────────────────────────────────────────────────────
let footFloorY = { left: Infinity, right: Infinity };

function resetFootFloor() { footFloorY = { left: Infinity, right: Infinity }; }

function enforceFootFloor(lAnkle, rAnkle, isGrounded = true) {
  if (!isGrounded) return { lAnkle, rAnkle };
  footFloorY.left = Math.min(footFloorY.left, lAnkle[1]);
  footFloorY.right = Math.min(footFloorY.right, rAnkle[1]);
  const LIFT_EPSILON = 0.5;
  return {
    lAnkle: [lAnkle[0], Math.max(lAnkle[1], footFloorY.left - LIFT_EPSILON), lAnkle[2]],
    rAnkle: [rAnkle[0], Math.max(rAnkle[1], footFloorY.right - LIFT_EPSILON), rAnkle[2]],
  };
}

// ── Place knee correctly ───────────────────────────────────────────────────────
function constrainKnee(hip, rawKnee, ankle, heel, toe, boneLenThigh, boneLenShin, pelvisFwd) {
  const rawThighDir = norm(sub(rawKnee, hip));
  const footVec = sub(toe, heel);
  let hingeAxis;
  
  if (len(footVec) > 0.001) {
    const footAlongThigh = dot(footVec, rawThighDir);
    const footPerp = norm(sub(footVec, scale(rawThighDir, footAlongThigh)));
    if (len(footPerp) > 0.001) {
      hingeAxis = norm(cross(rawThighDir, footPerp));
    }
  }
  if (!hingeAxis) hingeAxis = norm(cross(rawThighDir, [0, 0, 1]));

  const hipToAnkle = sub(ankle, hip);
  const dist = len(hipToAnkle);
  const totalLen = boneLenThigh + boneLenShin;

  if (dist >= totalLen) {
    const toAnkleDir = norm(hipToAnkle);
    const alongHinge = dot(toAnkleDir, hingeAxis);
    const inPlane = norm(sub(toAnkleDir, scale(hingeAxis, alongHinge)));
    return add(hip, scale(inPlane, boneLenThigh));
  }

  const d = Math.max(dist, 0.001);
  const cosHipAngle = Math.max(-1, Math.min(1, (boneLenThigh ** 2 + d ** 2 - boneLenShin ** 2) / (2 * boneLenThigh * d)));
  const hipAngle = Math.acos(cosHipAngle);

  const cosKneeAngle = Math.max(-1, Math.min(1, (boneLenThigh ** 2 + boneLenShin ** 2 - d ** 2) / (2 * boneLenThigh * boneLenShin)));
  const bendDeg = 180 - Math.acos(cosKneeAngle) * (180 / Math.PI);
  let effectiveHipAngle = hipAngle;
  
  if (bendDeg > 150) {
    const clampedKneeAngle = 30 * Math.PI / 180;
    const sinH = Math.sin(clampedKneeAngle) * boneLenShin / d;
    effectiveHipAngle = Math.asin(Math.max(-1, Math.min(1, sinH)));
  }

  const ankleAlongH = dot(hipToAnkle, hingeAxis);
  const hipToAnkleIP = sub(hipToAnkle, scale(hingeAxis, ankleAlongH));
  const planeFwd = len(hipToAnkleIP) > 0.001 ? norm(hipToAnkleIP) : norm(cross(hingeAxis, [0, 1, 0]));

  let planeSide = norm(cross(hingeAxis, planeFwd));
  const rawKneeOffset = sub(rawKnee, hip);
  const rawKneeAlongH = dot(rawKneeOffset, hingeAxis);
  const rawKneeInPlane = norm(sub(rawKneeOffset, scale(hingeAxis, rawKneeAlongH)));
  
  const rawSideComponent = dot(rawKneeInPlane, planeSide);
  if (rawSideComponent < 0) planeSide = scale(planeSide, -1);
  if (dot(planeSide, pelvisFwd) > 0) planeSide = scale(planeSide, -1);

  const kneeDir = add(scale(planeFwd, Math.cos(effectiveHipAngle)), scale(planeSide, Math.sin(effectiveHipAngle)));
  return add(hip, scale(norm(kneeDir), boneLenThigh));
}

// ── Elbow: use raw MediaPipe position ─────────────────────────────────────────
function constrainElbow(shoulder, rawElbow, wrist, boneLenUpper, boneLenFore) {
  const upperDir = norm(sub(rawElbow, shoulder));
  return add(shoulder, scale(upperDir, boneLenUpper));
}

// ── Extract and constrain all joint positions from a frame ────────────────────
function P(lms, worldLms, yawDeg, gravityAngleDeg = 0, geminiOrientation = null, geminiHandPoses = null) {
  const w = worldLms;

  const leftHip = mp(lms, w, 23, yawDeg, gravityAngleDeg);
  const rightHip = mp(lms, w, 24, yawDeg, gravityAngleDeg);
  const leftSho = mp(lms, w, 11, yawDeg, gravityAngleDeg);
  const rightSho = mp(lms, w, 12, yawDeg, gravityAngleDeg);
  const hips = avg(leftHip, rightHip);
  const shoulders = avg(leftSho, rightSho);
  const spine = avg(hips, shoulders);
  const spine1 = [(hips[0] + shoulders[0] * 2) / 3, (hips[1] + shoulders[1] * 2) / 3, (hips[2] + shoulders[2] * 2) / 3];
  const spine2 = shoulders;
  const leftEar = mp(lms, w, 7, yawDeg, gravityAngleDeg);
  const rightEar = mp(lms, w, 8, yawDeg, gravityAngleDeg);
  const earMid = avg(leftEar, rightEar);

  const hipRight = norm(sub(rightHip, leftHip));
  const spineUp = norm(sub(avg(leftSho, rightSho), avg(leftHip, rightHip)));
  const spineUpOrtho = norm(sub(spineUp, scale(hipRight, dot(spineUp, hipRight))));
  const pelvisFwd = norm(cross(hipRight, spineUpOrtho));

  let lAnkleRaw = mp(lms, w, 27, yawDeg, gravityAngleDeg);
  let rAnkleRaw = mp(lms, w, 28, yawDeg, gravityAngleDeg);
  let lWrist = mp(lms, w, 15, yawDeg, gravityAngleDeg);
  let rWrist = mp(lms, w, 16, yawDeg, gravityAngleDeg);

  let rawLKnee = mp(lms, w, 25, yawDeg, gravityAngleDeg);
  let rawRKnee = mp(lms, w, 26, yawDeg, gravityAngleDeg);
  let rawLElbow = mp(lms, w, 13, yawDeg, gravityAngleDeg);
  let rawRElbow = mp(lms, w, 14, yawDeg, gravityAngleDeg);

  if (cachedLengths) {
    const DOWN = [0, -1, 0];
    const OUT_L = [-1, -0.5, 0];
    const OUT_R = [1, -0.5, 0];

    if (len(rawLKnee) < 0.001) rawLKnee = add(leftHip, scale(DOWN, cachedLengths.lThigh));
    if (len(rawRKnee) < 0.001) rawRKnee = add(rightHip, scale(DOWN, cachedLengths.rThigh));
    if (len(lAnkleRaw) < 0.001) lAnkleRaw = add(rawLKnee, scale(DOWN, cachedLengths.lShin));
    if (len(rAnkleRaw) < 0.001) rAnkleRaw = add(rawRKnee, scale(DOWN, cachedLengths.rShin));

    if (len(rawLElbow) < 0.001) rawLElbow = add(leftSho, scale(norm(OUT_L), cachedLengths.lUpper));
    if (len(rawRElbow) < 0.001) rawRElbow = add(rightSho, scale(norm(OUT_R), cachedLengths.rUpper));
    if (len(lWrist) < 0.001) lWrist = add(rawLElbow, scale(pelvisFwd, cachedLengths.lFore));
    if (len(rWrist) < 0.001) rWrist = add(rawRElbow, scale(pelvisFwd, cachedLengths.rFore));
  }

  const isGrounded = geminiOrientation?.is_grounded ?? true;
  const { lAnkle, rAnkle } = enforceFootFloor(lAnkleRaw, rAnkleRaw, isGrounded);

  const lHeel = mp(lms, w, 29, yawDeg, gravityAngleDeg);
  const rHeel = mp(lms, w, 30, yawDeg, gravityAngleDeg);
  const lToe = mp(lms, w, 31, yawDeg, gravityAngleDeg);
  const rToe = mp(lms, w, 32, yawDeg, gravityAngleDeg);

  const bl = getBoneLengths(rawLKnee, rawRKnee, lAnkle, rAnkle, rawLElbow, rawRElbow, lWrist, rWrist, leftHip, rightHip, leftSho, rightSho);

  const leftKnee = constrainKnee(leftHip, rawLKnee, lAnkle, lHeel, lToe, bl.lThigh, bl.lShin, pelvisFwd);
  const rightKnee = constrainKnee(rightHip, rawRKnee, rAnkle, rHeel, rToe, bl.rThigh, bl.rShin, pelvisFwd);

  const leftElbow = constrainElbow(leftSho, rawLElbow, lWrist, bl.lUpper, bl.lFore);
  const rightElbow = constrainElbow(rightSho, rawRElbow, rWrist, bl.rUpper, bl.rFore);

  const shoulderWidth = len(sub(rightSho, leftSho));
  let minHandSep = shoulderWidth * 0.18;

  const handPoses = geminiHandPoses?.hands ?? [];
  const handsClose = handPoses.some(h => ['fist', 'clasp', 'other'].includes(h.pose) && h.confidence > 0.5);
  if (handsClose) {
    minHandSep = shoulderWidth * 0.04;
  }

  const wristVec = sub(rWrist, lWrist);
  const wristDist = len(wristVec);
  let adjLWrist = lWrist, adjRWrist = rWrist;
  if (wristDist < minHandSep && wristDist > 0.001) {
    const deficit = (minHandSep - wristDist) / 2;
    const pushDir = norm(sub(rightSho, leftSho));
    adjLWrist = sub(lWrist, scale(pushDir, deficit));
    adjRWrist = add(rWrist, scale(pushDir, deficit));
  }

  const lFootVec = sub(lToe, lHeel);
  const lShinDir = norm(sub(lAnkle, leftKnee));
  const lFootLateral = len(lFootVec) > 0.001 ? Math.abs(dot(norm(lFootVec), norm(cross(lShinDir, spineUpOrtho)))) : 0;
  const lFootRollLimit = Math.max(20, Math.min(40, 20 + lFootLateral * 40));
  const lFootTwistLimit = Math.max(25, Math.min(50, 25 + lFootLateral * 50));

  const rFootVec = sub(rToe, rHeel);
  const rShinDir = norm(sub(rAnkle, rightKnee));
  const rFootLateral = len(rFootVec) > 0.001 ? Math.abs(dot(norm(rFootVec), norm(cross(rShinDir, spineUpOrtho)))) : 0;
  const rFootRollLimit = Math.max(20, Math.min(40, 20 + rFootLateral * 40));
  const rFootTwistLimit = Math.max(25, Math.min(50, 25 + rFootLateral * 50));

  return {
    hips, spine, spine1, spine2,
    neck: avg(shoulders, earMid),
    head: earMid,
    leftShoulder: leftSho,
    leftArm: leftSho,
    leftForeArm: leftElbow,
    leftHand: adjLWrist,
    rightShoulder: rightSho,
    rightArm: rightSho,
    rightForeArm: rightElbow,
    rightHand: adjRWrist,
    leftUpLeg: leftHip,
    leftLeg: leftKnee,
    leftFoot: lAnkle,
    leftToeBase: mp(lms, w, 31, yawDeg, gravityAngleDeg),
    rightUpLeg: rightHip,
    rightLeg: rightKnee,
    rightFoot: rAnkle,
    rightToeBase: mp(lms, w, 32, yawDeg, gravityAngleDeg),
    lFootRollLimit, lFootTwistLimit,
    rFootRollLimit, rFootTwistLimit,
  };
}

// ── Rotation maths ────────────────────────────────────────────────────────────
function quatFromTo(from, to) {
  const f = norm(from), t = norm(to);
  const d = dot(f, t);
  if (d >= 1.0 - 1e-6) return [1, 0, 0, 0];
  if (d <= -1.0 + 1e-6) {
    let perp = cross(f, [1, 0, 0]);
    if (dot(perp, perp) < 1e-6) perp = cross(f, [0, 1, 0]);
    const ax = norm(perp);
    return [0, ax[0], ax[1], ax[2]];
  }
  const axis = cross(f, t);
  const w = Math.sqrt((1 + d) / 2);
  const s = 1 / (2 * w);
  return [w, axis[0] * s, axis[1] * s, axis[2] * s];
}

function quatMul([w1, x1, y1, z1], [w2, x2, y2, z2]) {
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}
const quatConj = ([w, x, y, z]) => [w, -x, -y, -z];
function quatRotate([w, x, y, z], [vx, vy, vz]) {
  const q = [w, x, y, z], qv = [0, vx, vy, vz];
  const [, rx, ry, rz] = quatMul(quatMul(q, qv), quatConj(q));
  return [rx, ry, rz];
}
function quatToZXY([w, x, y, z]) {
  const m = [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
  const rx = Math.asin(Math.max(-1, Math.min(1, m[7])));
  const ry = Math.atan2(-m[6], m[8]);
  const rz = Math.atan2(-m[1], m[4]);
  const deg = r => r * (180 / Math.PI);
  return [deg(rz), deg(rx), deg(ry)];
}

export { P, getWristFlex, wristRotationDeg, resolveFingerAngles, resetBoneLengthCache, resetFootFloor };


// ── Rest offsets ──────────────────────────────────────────────────────────────
function getRestOffsets() {
  return {
    hips:          [0,   0,  0], spine:         [0,  10,  0],
    spine1:        [0,  10,  0], spine2:        [0,  10,  0],
    neck:          [0,   8,  0], head:          [0,   8,  0],
    headEnd:       [0,   8,  0],
    leftShoulder:  [-8,  0,  0], leftArm:       [0,   0,  0],
    leftForeArm:   [-15, 0,  0], leftHand:      [-12, 0,  0],
    leftHandEnd:   [-12, 0,  0],
    rightShoulder: [8,   0,  0], rightArm:      [0,   0,  0],
    rightForeArm:  [15,  0,  0], rightHand:     [12,  0,  0],
    rightHandEnd:  [12,  0,  0],
    leftUpLeg:     [-8, -20, 0], leftLeg:       [0,  -20, 0],
    leftFoot:      [0,  -18, 0], leftToeBase:   [0,   0,  5],
    leftToeEnd:    [0,   0,  5],
    rightUpLeg:    [8,  -20, 0], rightLeg:      [0,  -20, 0],
    rightFoot:     [0,  -18, 0], rightToeBase:  [0,   0,  5],
    rightToeEnd:   [0,   0,  5],
    // ── Finger rest offsets (in hand-local space, along Z = finger extension direction)
    // Left hand fingers extend in -X direction; right hand in +X
    // Thumb sits offset slightly up (+Y) and to the side
    lThumb1:   [-1.5, 0.5, 1.5],  lThumb2:   [0, 0, 3.0],  lThumb3:   [0, 0, 2.5],  lThumbEnd:   [0, 0, 2.0],
    lIndex1:   [-1.0, 0,   4.0],  lIndex2:   [0, 0, 3.0],  lIndex3:   [0, 0, 2.0],  lIndexEnd:   [0, 0, 1.5],
    lMiddle1:  [ 0.0, 0,   4.0],  lMiddle2:  [0, 0, 3.0],  lMiddle3:  [0, 0, 2.0],  lMiddleEnd:  [0, 0, 1.5],
    lRing1:    [ 1.0, 0,   4.0],  lRing2:    [0, 0, 3.0],  lRing3:    [0, 0, 2.0],  lRingEnd:    [0, 0, 1.5],
    lPinky1:   [ 2.0, 0,   3.5],  lPinky2:   [0, 0, 2.5],  lPinky3:   [0, 0, 2.0],  lPinkyEnd:  [0, 0, 1.5],
    rThumb1:   [ 1.5, 0.5, 1.5],  rThumb2:   [0, 0, 3.0],  rThumb3:   [0, 0, 2.5],  rThumbEnd:   [0, 0, 2.0],
    rIndex1:   [ 1.0, 0,   4.0],  rIndex2:   [0, 0, 3.0],  rIndex3:   [0, 0, 2.0],  rIndexEnd:   [0, 0, 1.5],
    rMiddle1:  [ 0.0, 0,   4.0],  rMiddle2:  [0, 0, 3.0],  rMiddle3:  [0, 0, 2.0],  rMiddleEnd:  [0, 0, 1.5],
    rRing1:    [-1.0, 0,   4.0],  rRing2:    [0, 0, 3.0],  rRing3:    [0, 0, 2.0],  rRingEnd:    [0, 0, 1.5],
    rPinky1:   [-2.0, 0,   3.5],  rPinky2:   [0, 0, 2.5],  rPinky3:   [0, 0, 2.0],  rPinkyEnd:  [0, 0, 1.5],
  }
}

const f = n => n.toFixed(4)
const o = ([x,y,z]) => `${f(x)} ${f(y)} ${f(z)}`
const t = n => '\t'.repeat(n)

// ── Build finger sub-hierarchy for one hand ────────────────────────────────────
// side: 'l' or 'r'   prefix: 'Left' or 'Right'   depth: tab depth
function buildFingerHierarchy(side, prefix, depth, off) {
  const d = depth
  const s = side  // 'l' or 'r'

  function joint(name, offKey, channels, children) {
    const indent = t(d)
    return [
      `${indent}JOINT ${prefix}Hand${name}`,
      `${indent}{`,
      `${indent}\tOFFSET ${o(off[`${s}${offKey}`])}`,
      `${indent}\tCHANNELS ${channels}`,
      ...children,
      `${indent}}`,
    ].join('\n')
  }

  function finger(fingerName, offPrefix) {
    // Three joints + end site
    return joint(`${fingerName}1`, `${offPrefix}1`, '3 Zrotation Xrotation Yrotation', [
      joint(`${fingerName}2`, `${offPrefix}2`, '3 Zrotation Xrotation Yrotation', [
        joint(`${fingerName}3`, `${offPrefix}3`, '3 Zrotation Xrotation Yrotation', [
          `${t(d+3)}End Site`,
          `${t(d+3)}{`,
          `${t(d+4)}OFFSET ${o(off[`${s}${offPrefix}End`])}`,
          `${t(d+3)}}`,
        ]),
      ]),
    ])
  }

  return [
    finger('Thumb',  'Thumb'),
    finger('Index',  'Index'),
    finger('Middle', 'Middle'),
    finger('Ring',   'Ring'),
    finger('Pinky',  'Pinky'),
  ].join('\n')
}

function buildHierarchy(off) {
  // Build finger sections
  const lFingers = buildFingerHierarchy('l', 'Left',  8, off)
  const rFingers = buildFingerHierarchy('r', 'Right', 8, off)

  return `HIERARCHY
ROOT Hips
{
${t(1)}OFFSET ${o(off.hips)}
${t(1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
${t(1)}JOINT Spine
${t(1)}{
${t(2)}OFFSET ${o(off.spine)}
${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(2)}JOINT Spine1
${t(2)}{
${t(3)}OFFSET ${o(off.spine1)}
${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(3)}JOINT Spine2
${t(3)}{
${t(4)}OFFSET ${o(off.spine2)}
${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(4)}JOINT Neck
${t(4)}{
${t(5)}OFFSET ${o(off.neck)}
${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(5)}JOINT Head
${t(5)}{
${t(6)}OFFSET ${o(off.head)}
${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(6)}End Site
${t(6)}{
${t(7)}OFFSET ${o(off.headEnd)}
${t(6)}}
${t(5)}}
${t(4)}}
${t(4)}JOINT LeftShoulder
${t(4)}{
${t(5)}OFFSET ${o(off.leftShoulder)}
${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(5)}JOINT LeftArm
${t(5)}{
${t(6)}OFFSET ${o(off.leftArm)}
${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(6)}JOINT LeftForeArm
${t(6)}{
${t(7)}OFFSET ${o(off.leftForeArm)}
${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(7)}JOINT LeftHand
${t(7)}{
${t(8)}OFFSET ${o(off.leftHand)}
${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation
${lFingers}
${t(7)}}
${t(6)}}
${t(5)}}
${t(4)}}
${t(4)}JOINT RightShoulder
${t(4)}{
${t(5)}OFFSET ${o(off.rightShoulder)}
${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(5)}JOINT RightArm
${t(5)}{
${t(6)}OFFSET ${o(off.rightArm)}
${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(6)}JOINT RightForeArm
${t(6)}{
${t(7)}OFFSET ${o(off.rightForeArm)}
${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(7)}JOINT RightHand
${t(7)}{
${t(8)}OFFSET ${o(off.rightHand)}
${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation
${rFingers}
${t(7)}}
${t(6)}}
${t(5)}}
${t(4)}}
${t(3)}}
${t(2)}}
${t(1)}}
${t(1)}JOINT LeftUpLeg
${t(1)}{
${t(2)}OFFSET ${o(off.leftUpLeg)}
${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(2)}JOINT LeftLeg
${t(2)}{
${t(3)}OFFSET ${o(off.leftLeg)}
${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(3)}JOINT LeftFoot
${t(3)}{
${t(4)}OFFSET ${o(off.leftFoot)}
${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(4)}JOINT LeftToeBase
${t(4)}{
${t(5)}OFFSET ${o(off.leftToeBase)}
${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(5)}End Site
${t(5)}{
${t(6)}OFFSET ${o(off.leftToeEnd)}
${t(5)}}
${t(4)}}
${t(3)}}
${t(2)}}
${t(1)}}
${t(1)}JOINT RightUpLeg
${t(1)}{
${t(2)}OFFSET ${o(off.rightUpLeg)}
${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(2)}JOINT RightLeg
${t(2)}{
${t(3)}OFFSET ${o(off.rightLeg)}
${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(3)}JOINT RightFoot
${t(3)}{
${t(4)}OFFSET ${o(off.rightFoot)}
${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(4)}JOINT RightToeBase
${t(4)}{
${t(5)}OFFSET ${o(off.rightToeBase)}
${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation
${t(5)}End Site
${t(5)}{
${t(6)}OFFSET ${o(off.rightToeEnd)}
${t(5)}}
${t(4)}}
${t(3)}}
${t(2)}}
${t(1)}}
}`
}

// ── Rear-view correction ──────────────────────────────────────────────────────
// When the camera is behind the subject, MediaPipe's x coordinates are mirrored and left/right joints are swapped from the body's perspective.
// With yaw pre-rotation active, this only fires for true rear-facing views where pelvisFwd[2] < 0 after rotation — i.e. Gemini yaw is near ±180°.
function correctRearView(p, pelvisFwd) {
  if (pelvisFwd[2] >= 0) return p // Front-facing after rotation, no correction needed

  // Mirror x for all joint positions, and swap left/right pairs
  const mx = ([x, y, z]) => [-x, y, z]

  return {
    hips:          mx(p.hips),
    spine:         mx(p.spine),
    spine1:        mx(p.spine1),
    spine2:        mx(p.spine2),
    neck:          mx(p.neck),
    head:          mx(p.head),
    leftShoulder:  mx(p.rightShoulder),
    leftArm:       mx(p.rightArm),
    leftForeArm:   mx(p.rightForeArm),
    leftHand:      mx(p.rightHand),
    rightShoulder: mx(p.leftShoulder),
    rightArm:      mx(p.leftArm),
    rightForeArm:  mx(p.leftForeArm),
    rightHand:     mx(p.leftHand),
    leftUpLeg:     mx(p.rightUpLeg),
    leftLeg:       mx(p.rightLeg),
    leftFoot:      mx(p.rightFoot),
    leftToeBase:   mx(p.rightToeBase),
    rightUpLeg:    mx(p.leftUpLeg),
    rightLeg:      mx(p.leftLeg),
    rightFoot:     mx(p.leftFoot),
    rightToeBase:  mx(p.leftToeBase),
    lFootRollLimit:  p.rFootRollLimit,
    lFootTwistLimit: p.rFootTwistLimit,
    rFootRollLimit:  p.lFootRollLimit,
    rFootTwistLimit: p.lFootTwistLimit,
  }
}

// ── Emit finger rotation channels for one hand ────────────────────────────────
// fingerAngles: { thumb:{mcp,ip}, index:{mcp,pip,dip}, middle, ring, pinky }
// Each joint emits 3 channels (Zrotation Xrotation Yrotation).
// Bend maps to Xrotation (flexion axis). Z/Y stay 0 except thumb which gets slight Z spread.
function emitFingerRotations(fingerAngles, side) {
  const vals = []

  if (!fingerAngles) {
    // No data — emit 45 zeros (5 fingers × 3 joints × 3 channels)
    for (let i = 0; i < 45; i++) vals.push(0)
    return vals
  }

  // Clamp bend to [0, 90]
  const bend = (deg) => Math.max(0, Math.min(90, deg ?? 0))
  const sign = side === 'left' ? 1 : 1  // both positive in Xrot for BVH flexion

  const { thumb, index, middle, ring, pinky } = fingerAngles

  // Thumb (2 joints: MCP, IP) — slight Z spread to pull it away from palm
  const thumbSpread = side === 'left' ? -15 : 15
  // Joint 1 (MCP)
  vals.push(thumbSpread, bend(thumb?.mcp) * sign, 0)
  // Joint 2 (IP)
  vals.push(0, bend(thumb?.ip) * sign, 0)
  // Joint 3 — thumb tip, minimal movement
  vals.push(0, bend(thumb?.ip) * 0.3 * sign, 0)

  // Index (3 joints: MCP, PIP, DIP)
  vals.push(0, bend(index?.mcp) * sign, 0)
  vals.push(0, bend(index?.pip) * sign, 0)
  vals.push(0, bend(index?.dip) * sign, 0)

  // Middle
  vals.push(0, bend(middle?.mcp) * sign, 0)
  vals.push(0, bend(middle?.pip) * sign, 0)
  vals.push(0, bend(middle?.dip) * sign, 0)

  // Ring
  vals.push(0, bend(ring?.mcp) * sign, 0)
  vals.push(0, bend(ring?.pip) * sign, 0)
  vals.push(0, bend(ring?.dip) * sign, 0)

  // Pinky
  vals.push(0, bend(pinky?.mcp) * sign, 0)
  vals.push(0, bend(pinky?.pip) * sign, 0)
  vals.push(0, bend(pinky?.dip) * sign, 0)

  return vals
}

function buildMotion(frames, frameTime, off, captureFps) {
  const lines = ['MOTION', `Frames: ${frames.length}`, `Frame Time: ${f(frameTime)}`]
  const DOWN = [0, -1, 0], FWD = [0, 0, 1]
  const REST = {
    spine: norm(off.spine), spine1: norm(off.spine1), spine2: norm(off.spine2),
    neck: norm(off.neck), head: norm(off.head),
    leftShoulder: norm(off.leftShoulder), leftForeArm: norm(off.leftForeArm), leftHand: norm(off.leftHand),
    rightShoulder: norm(off.rightShoulder), rightArm: norm(off.rightArm), rightForeArm: norm(off.rightForeArm), rightHand: norm(off.rightHand),
    leftUpLeg: DOWN, leftLeg: DOWN, leftFoot: DOWN, leftToeBase: FWD,
    rightUpLeg: DOWN, rightLeg: DOWN, rightFoot: DOWN, rightToeBase: FWD,
  }

  const orientEst = new OrientationEstimator({ captureFps })

  for (const frame of frames) {
    // ── Orientation estimation ─────────────────────────────────────────
    const enrichedFrame   = orientEst.process(frame)
    const { orientation } = enrichedFrame
    const yawDeg          = wrapYaw(orientation.yaw)

    if (orientation.shotCut) {
      console.log(`[BVH] Shot cut at frame ${frame.frameIndex}, yaw snapped to ${yawDeg.toFixed(1)}°`)
    }
    if (orientation.source === 'gemini') {
      console.log(`[BVH] Gemini orientation: ${orientation.view} (${yawDeg.toFixed(1)}°)`)
    }

    const gravityData    = frame.geminiGravityOrientation
    const gravityAngleDeg = (gravityData?.confidence > 0.6 && gravityData?.gravity_angle !== undefined)
      ? gravityData.gravity_angle
      : 0

    const pRaw = P(
      frame.landmarks,
      frame.worldLandmarks,
      yawDeg,
      gravityAngleDeg,
      frame.geminiGravityOrientation,
      frame.geminiHandPoses
    )

    const lh = pRaw.leftUpLeg, rh = pRaw.rightUpLeg
    const ls = pRaw.leftShoulder, rs = pRaw.rightShoulder
    const hipRight     = norm(sub(rh, lh))
    const spineUp      = norm(sub(avg(ls, rs), avg(lh, rh)))
    const spineUpOrtho = norm(sub(spineUp, scale(hipRight, dot(spineUp, hipRight))))
    const pelvisFwd    = norm(cross(hipRight, spineUpOrtho))
    const p            = correctRearView(pRaw, pelvisFwd)

    const vals = []

    // ── Hips ───────────────────────────────────────────────────────────
    vals.push(...p.hips)
    const spineDir = sub(p.spine, p.hips)
    const hipsRot  = quatFromTo(REST.spine, norm(spineDir))
    vals.push(...quatToZXY(hipsRot))

    // ── Spine chain ────────────────────────────────────────────────────
    const spine1Dir   = sub(p.spine1, p.spine)
    const spineRot    = quatFromTo(quatRotate(hipsRot, REST.spine1), norm(spine1Dir))
    const spineLocal  = quatMul(quatConj(hipsRot), quatMul(spineRot, hipsRot))
    vals.push(...quatToZXY(spineLocal))

    const spine2Dir   = sub(p.spine2,  p.spine1)
    const spineWorld1 = quatMul(hipsRot, spineLocal)
    const spine1Rot   = quatFromTo(quatRotate(spineWorld1, REST.spine2), norm(spine2Dir))
    const spine1Local = quatMul(quatConj(spineWorld1), quatMul(spine1Rot, spineWorld1))
    vals.push(...quatToZXY(spine1Local))

    const spineWorld2Base = quatMul(spineWorld1, spine1Local)
    const actualShoVec    = norm(sub(p.rightShoulder, p.leftShoulder))
    const expectedShoVec  = quatRotate(spineWorld2Base, norm(off.rightShoulder))
    const twistQuat       = quatFromTo(
      norm([expectedShoVec[0], 0, expectedShoVec[2]]),
      norm([actualShoVec[0],   0, actualShoVec[2]])
    )
    const neckDir_        = sub(p.neck, p.spine2)
    const spine2DirRot    = quatFromTo(quatRotate(spineWorld2Base, REST.neck), norm(neckDir_))
    const spine2WithTwist = quatMul(twistQuat, spine2DirRot)
    const spine2Local     = quatMul(quatConj(spineWorld2Base), quatMul(spine2WithTwist, spineWorld2Base))
    vals.push(...quatToZXY(spine2Local))

    const neckDir   = sub(p.head, p.neck)
    const neckWorld = quatMul(spineWorld2Base, spine2Local)
    const neckRot   = quatFromTo(quatRotate(neckWorld, REST.head), norm(neckDir))
    const neckLocal = quatMul(quatConj(neckWorld), quatMul(neckRot, neckWorld))
    vals.push(...quatToZXY(neckLocal))
    const spineWorld2 = neckWorld
    vals.push(0, 0, 0)

    // ── Left arm ───────────────────────────────────────────────────────
    const lShoDir   = sub(p.leftShoulder, p.spine2)
    const lShoRot   = quatFromTo(quatRotate(spineWorld2, REST.leftShoulder), norm(lShoDir))
    const lShoLocal = quatMul(quatConj(spineWorld2), quatMul(lShoRot, spineWorld2))
    vals.push(...quatToZXY(lShoLocal))

    const lArmDir   = sub(p.leftForeArm, p.leftShoulder)
    const lShoWorld = quatMul(spineWorld2, lShoLocal)
    const lArmRot   = quatFromTo(quatRotate(lShoWorld, REST.leftForeArm), norm(lArmDir))
    const lArmLocal = quatMul(quatConj(lShoWorld), quatMul(lArmRot, lShoWorld))
    vals.push(...quatToZXY(lArmLocal))

    const lFADir    = sub(p.leftHand, p.leftForeArm)
    const lArmWorld = quatMul(lShoWorld, lArmLocal)
    const lFARot    = quatFromTo(quatRotate(lArmWorld, REST.leftHand), norm(lFADir))
    const lFALocal  = quatMul(quatConj(lArmWorld), quatMul(lFARot, lArmWorld))
    vals.push(...quatToZXY(lFALocal))

    // ── Left Hand (Fixed Orientation Offset) ───────────────────────────
    const lHandGemini = frame.geminiHandPoses?.hands?.find(h => h.side === 'left' && h.confidence > 0.5)
    const lWristRot   = lHandGemini ? wristRotationDeg(lHandGemini.wrist_rotation, 'left') : 0
    const lWristFlex  = getWristFlex(frame, 'left', lFADir, yawDeg)
    
    // Counteract the 90-degree twist caused by finger tracking Z-forward offsets
    vals.push(lWristRot, lWristFlex, -90) 

    const lFingerAngles = resolveFingerAngles(frame, 'left')
    vals.push(...emitFingerRotations(lFingerAngles, 'left'))

    // ── Right arm ──────────────────────────────────────────────────────
    const rShoDir   = sub(p.rightShoulder, p.spine2)
    const rShoRot   = quatFromTo(quatRotate(spineWorld2, REST.rightShoulder), norm(rShoDir))
    const rShoLocal = quatMul(quatConj(spineWorld2), quatMul(rShoRot, spineWorld2))
    vals.push(...quatToZXY(rShoLocal))

    const rArmDir   = sub(p.rightForeArm, p.rightShoulder)
    const rShoWorld = quatMul(spineWorld2, rShoLocal)
    const rArmRot   = quatFromTo(quatRotate(rShoWorld, REST.rightForeArm), norm(rArmDir))
    const rArmLocal = quatMul(quatConj(rShoWorld), quatMul(rArmRot, rShoWorld))
    vals.push(...quatToZXY(rArmLocal))

    const rFADir    = sub(p.rightHand, p.rightForeArm)
    const rArmWorld = quatMul(rShoWorld, rArmLocal)
    const rFARot    = quatFromTo(quatRotate(rArmWorld, REST.rightHand), norm(rFADir))
    const rFALocal  = quatMul(quatConj(rArmWorld), quatMul(rFARot, rArmWorld))
    vals.push(...quatToZXY(rFALocal))

    // ── Right Hand (Fixed Orientation Offset) ──────────────────────────
    const rHandGemini = frame.geminiHandPoses?.hands?.find(h => h.side === 'right' && h.confidence > 0.5)
    const rWristRot   = rHandGemini ? wristRotationDeg(rHandGemini.wrist_rotation, 'right') : 0
    const rWristFlex  = getWristFlex(frame, 'right', rFADir, yawDeg)
    
    // Balance right hand coordinate frame projection parity
    vals.push(rWristRot, rWristFlex, 90) 

    const rFingerAngles = resolveFingerAngles(frame, 'right')
    vals.push(...emitFingerRotations(rFingerAngles, 'right'))

    // ── Left leg ───────────────────────────────────────────────────────
    const lULDir   = sub(p.leftLeg, p.leftUpLeg)
    const lULRot   = quatFromTo(quatRotate(hipsRot, REST.leftUpLeg), norm(lULDir))
    const lULLocal = quatMul(quatConj(hipsRot), quatMul(lULRot, hipsRot))
    vals.push(...quatToZXY(lULLocal))
    const lULWorld = quatMul(hipsRot, lULLocal)

    const lLDir   = sub(p.leftFoot, p.leftLeg)
    const lLRot   = quatFromTo(quatRotate(lULWorld, REST.leftLeg), norm(lLDir))
    const lLLocal = quatMul(quatConj(lULWorld), quatMul(lLRot, lULWorld))
    vals.push(...quatToZXY(lLLocal))
    const lLWorld = quatMul(lULWorld, lLLocal)

    const lFDir   = sub(p.leftToeBase, p.leftFoot)
    const lFRot   = quatFromTo(quatRotate(lLWorld, REST.leftFoot), norm(lFDir))
    const lFLocal = quatMul(quatConj(lLWorld), quatMul(lFRot, lLWorld))
    let lFEuler   = quatToZXY(lFLocal)
    lFEuler[0]    = Math.max(-p.lFootRollLimit,  Math.min(p.lFootRollLimit,  lFEuler[0]))
    lFEuler[2]    = Math.max(-p.lFootTwistLimit, Math.min(p.lFootTwistLimit, lFEuler[2]))
    vals.push(...lFEuler)
    vals.push(0, 0, 0)

    // ── Right leg ──────────────────────────────────────────────────────
    const rULDir   = sub(p.rightLeg, p.rightUpLeg)
    const rULRot   = quatFromTo(quatRotate(hipsRot, REST.rightUpLeg), norm(rULDir))
    const rULLocal = quatMul(quatConj(hipsRot), quatMul(rULRot, hipsRot))
    vals.push(...quatToZXY(rULLocal))
    const rULWorld = quatMul(hipsRot, rULLocal)

    const rLDir   = sub(p.rightFoot, p.rightLeg)
    const rLRot   = quatFromTo(quatRotate(rULWorld, REST.rightLeg), norm(rLDir))
    const rLLocal = quatMul(quatConj(rULWorld), quatMul(rLRot, rULWorld))
    vals.push(...quatToZXY(rLLocal))
    const rLWorld = quatMul(rULWorld, rLLocal)

    const rFDir   = sub(p.rightToeBase, p.rightFoot)
    const rFRot   = quatFromTo(quatRotate(rLWorld, REST.rightFoot), norm(rFDir))
    const rFLocal = quatMul(quatConj(rLWorld), quatMul(rFRot, rLWorld))
    let rFEuler   = quatToZXY(rFLocal)
    rFEuler[0]    = Math.max(-p.rFootRollLimit,  Math.min(p.rFootRollLimit,  rFEuler[0]))
    rFEuler[2]    = Math.max(-p.rFootTwistLimit, Math.min(p.rFootTwistLimit, rFEuler[2]))
    vals.push(...rFEuler)
    vals.push(0, 0, 0)

    lines.push(vals.map(v => f(v)).join(' '))
  }

  return lines.join('\n')
}

// ── Public ────────────────────────────────────────────────────────────────────
// clipName: optional snake_case string from Gemini's /name-clip endpoint.
// If provided, the downloaded file is named after the detected activity instead of "pose_sequence".
export function exportBVH(frames, captureFps, clipName = null) {
  if (!frames?.length) return
  console.log(`[BVH] Exporting ${frames.length} frames at ${captureFps} FPS`)
  resetBoneLengthCache()
  resetFootFloor()
  const off      = getRestOffsets()
  const bvh      = buildHierarchy(off) + '\n' + buildMotion(frames, 1 / captureFps, off, captureFps)
  const blob     = new Blob([bvh], { type: 'text/plain' })
  const url      = URL.createObjectURL(blob)
  const a        = document.createElement('a')
  const filename = clipName ? `${clipName}.bvh` : 'pose_sequence.bvh'
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
  console.log(`[BVH] Export complete → ${filename}`)
}

// ── Single-image export ────────────────────────────────────────────────────────
export function exportSingleImageBVH(landmark, worldLandmark, geminiOrientation, captureFps = 30, handData = null, geminiFingerPoses = null) {
  if (!landmark) return
  const frame = {
    frameIndex:        0,
    landmarks:         landmark,
    worldLandmarks:    worldLandmark,
    geminiOrientation: geminiOrientation,
    handData:          handData,
    geminiFingerPoses: geminiFingerPoses,
  }
  console.log(`[BVH] Exporting single image with orientation:`, geminiOrientation)
  exportBVH([frame], captureFps)
}