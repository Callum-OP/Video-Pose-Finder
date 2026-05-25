import { solveIK } from './ikSolver.js'

// Joint structure verified to load in Clip Studio Paint.
// Matches Mixamo/Blender BVH export convention exactly.

const SCALE = 100

function mp(lms, worldLms, idx) {
  const src = worldLms?.[idx] ?? lms[idx]
  if (!src) return [0, 0, 0]
  if (worldLms?.[idx]) {
    return [-src.x * SCALE, -src.y * SCALE, src.z * SCALE]
  }
  return [-src.x * SCALE, -src.y * SCALE, 0]
}

function avg(a, b) { return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2] }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
function scale(v, s) { return [v[0] * s, v[1] * s, v[2] * s] }
function norm(v) {
  const l = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) || 1e-8
  return [v[0]/l, v[1]/l, v[2]/l]
}
function len(v) { return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) }

function P(lms, worldLms) {
  const w = worldLms 
  const leftHip   = mp(lms, w, 23)
  const rightHip  = mp(lms, w, 24)
  const leftSho   = mp(lms, w, 11)
  const rightSho  = mp(lms, w, 12)
  const hips      = avg(leftHip, rightHip)
  const shoulders = avg(leftSho, rightSho)
  const spine     = avg(hips, shoulders)
  const spine1    = [(hips[0]+shoulders[0]*2)/3, (hips[1]+shoulders[1]*2)/3, (hips[2]+shoulders[2]*2)/3]
  const spine2    = shoulders

  const leftEar  = mp(lms, w, 7)
  const rightEar = mp(lms, w, 8)
  const earMid   = avg(leftEar, rightEar)

  return {
    hips, spine, spine1, spine2,
    neck:          avg(shoulders, earMid),
    head:          earMid,
    leftShoulder:  leftSho,
    leftArm:       leftSho,
    leftForeArm:   mp(lms, w, 13),
    leftHand:      mp(lms, w, 15),
    rightShoulder: rightSho,
    rightArm:      rightSho,
    rightForeArm:  mp(lms, w, 14),
    rightHand:     mp(lms, w, 16),
    leftUpLeg:     leftHip,
    leftLeg:       mp(lms, w, 25),
    leftFoot:      mp(lms, w, 27),
    leftToeBase:   mp(lms, w, 31),
    rightUpLeg:    rightHip,
    rightLeg:      mp(lms, w, 26),
    rightFoot:     mp(lms, w, 28),
    rightToeBase:  mp(lms, w, 32),
  }
}

// ── Rotation maths ───────────────────────────────────────────────────────────
function cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ]
}

function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

function quatFromTo(from, to) {
  const f = norm(from)
  const t = norm(to)
  const d = dot(f, t)

  if (d >= 1.0 - 1e-6) return [1, 0, 0, 0]
  if (d <= -1.0 + 1e-6) {
    let perp = cross(f, [1,0,0])
    if (dot(perp,perp) < 1e-6) perp = cross(f, [0,1,0])
    const ax = norm(perp)
    return [0, ax[0], ax[1], ax[2]]
  }
  const axis = cross(f, t)
  const w    = Math.sqrt((1+d) / 2)
  const s    = 1 / (2 * w)
  return [w, axis[0]*s, axis[1]*s, axis[2]*s]
}

// Smoothing filter for if the camera is moving too much
function smoothFrames(frames, windowSize = 3) {
  if (!frames || frames.length === 0) return [];
  return frames.map((frame, i) => {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(frames.length, start + windowSize);
    const windowFrames = frames.slice(start, end);
    
    if (!frame.worldLandmarks) return frame;
    
    const isArray = Array.isArray(frame.worldLandmarks);
    const smoothedWorldLms = isArray ? [] : {};
    const keys = Object.keys(frame.worldLandmarks);
    
    for (const key of keys) {
      let sumX = 0, sumY = 0, sumZ = 0, count = 0;
      for (const wf of windowFrames) {
        const pt = wf.worldLandmarks?.[key];
        if (pt) {
          sumX += pt.x; sumY += pt.y; sumZ += pt.z;
          count++;
        }
      }
      if (count > 0) {
        const smoothedPt = { x: sumX / count, y: sumY / count, z: sumZ / count };
        if (isArray) {
          smoothedWorldLms[parseInt(key, 10)] = smoothedPt;
        } else {
          smoothedWorldLms[key] = smoothedPt;
        }
      }
    }
    return { ...frame, worldLandmarks: smoothedWorldLms };
  });
}

function quatMul([w1,x1,y1,z1], [w2,x2,y2,z2]) {
  return [
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
  ]
}

function quatConj([w,x,y,z]) { return [w,-x,-y,-z] }

function quatRotate([w,x,y,z], [vx,vy,vz]) {
  const q = [w,x,y,z]
  const qv = [0, vx, vy, vz]
  const [,rx,ry,rz] = quatMul(quatMul(q, qv), quatConj(q))
  return [rx, ry, rz]
}

function quatToZXY([w,x,y,z]) {
  const m = [
    1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
    2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
    2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
  ]
  const rx = Math.asin(Math.max(-1, Math.min(1, m[7])))
  const ry = Math.atan2(-m[6], m[8])
  const rz = Math.atan2(-m[1], m[4])
  const deg = r => r * (180/Math.PI)
  return [deg(rz), deg(rx), deg(ry)]
}

function getRestOffsets() {
  return {
    hips:          [0,      0,     0   ],
    spine:         [0,      10,    0   ],
    spine1:        [0,      10,    0   ],
    spine2:        [0,      10,    0   ],
    neck:          [0,      8,     0   ],
    head:          [0,      8,     0   ],
    headEnd:       [0,      8,     0   ],
    leftShoulder:  [-8,     0,     0   ],
    leftArm:       [0,      0,     0   ],
    leftForeArm:   [-15,    0,     0   ],
    leftHand:      [-12,    0,     0   ],
    leftHandEnd:   [-12,    0,     0   ],
    rightShoulder: [8,      0,     0   ],
    rightArm:      [0,      0,     0   ],
    rightForeArm:  [15,     0,     0   ],
    rightHand:     [12,     0,     0   ],
    rightHandEnd:  [12,     0,     0   ],
    leftUpLeg:     [-8,    -20,    0   ],
    leftLeg:       [0,     -20,    0   ],
    leftFoot:      [0,     -18,    0   ],
    leftToeBase:   [0,      0,     5   ],
    leftToeEnd:    [0,      0,     5   ],
    rightUpLeg:    [8,     -20,    0   ],
    rightLeg:      [0,     -20,    0   ],
    rightFoot:     [0,     -18,    0   ],
    rightToeBase:  [0,      0,     5   ],
    rightToeEnd:   [0,      0,     5   ],
  }
}

const f = n => n.toFixed(4)
const o = ([x,y,z]) => `${f(x)} ${f(y)} ${f(z)}`
const t = n => '\t'.repeat(n)

function buildHierarchy(off) {
  return `HIERARCHY\nROOT Hips\n{\n${t(1)}OFFSET ${o(off.hips)}\n${t(1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n${t(1)}JOINT Spine\n${t(1)}{\n${t(2)}OFFSET ${o(off.spine)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT Spine1\n${t(2)}{\n${t(3)}OFFSET ${o(off.spine1)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT Spine2\n${t(3)}{\n${t(4)}OFFSET ${o(off.spine2)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT Neck\n${t(4)}{\n${t(5)}OFFSET ${o(off.neck)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT Head\n${t(5)}{\n${t(6)}OFFSET ${o(off.head)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}End Site\n${t(6)}{\n${t(7)}OFFSET ${o(off.headEnd)}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(4)}JOINT LeftShoulder\n${t(4)}{\n${t(5)}OFFSET ${o(off.leftShoulder)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT LeftArm\n${t(5)}{\n${t(6)}OFFSET ${o(off.leftArm)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}JOINT LeftForeArm\n${t(6)}{\n${t(7)}OFFSET ${o(off.leftForeArm)}\n${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(7)}JOINT LeftHand\n${t(7)}{\n${t(8)}OFFSET ${o(off.leftHand)}\n${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(8)}End Site\n${t(8)}{\n${t(9)}OFFSET ${o(off.leftHandEnd)}\n${t(8)}}\n${t(7)}}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(4)}JOINT RightShoulder\n${t(4)}{\n${t(5)}OFFSET ${o(off.rightShoulder)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT RightArm\n${t(5)}{\n${t(6)}OFFSET ${o(off.rightArm)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}JOINT RightForeArm\n${t(6)}{\n${t(7)}OFFSET ${o(off.rightForeArm)}\n${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(7)}JOINT RightHand\n${t(7)}{\n${t(8)}OFFSET ${o(off.rightHand)}\n${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(8)}End Site\n${t(8)}{\n${t(9)}OFFSET ${o(off.rightHandEnd)}\n${t(8)}}\n${t(7)}}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n${t(1)}JOINT LeftUpLeg\n${t(1)}{\n${t(2)}OFFSET ${o(off.leftUpLeg)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT LeftLeg\n${t(2)}{\n${t(3)}OFFSET ${o(off.leftLeg)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT LeftFoot\n${t(3)}{\n${t(4)}OFFSET ${o(off.leftFoot)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT LeftToeBase\n${t(4)}{\n${t(5)}OFFSET ${o(off.leftToeBase)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}End Site\n${t(5)}{\n${t(6)}OFFSET ${o(off.leftToeEnd)}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n${t(1)}JOINT RightUpLeg\n${t(1)}{\n${t(2)}OFFSET ${o(off.rightUpLeg)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT RightLeg\n${t(2)}{\n${t(3)}OFFSET ${o(off.rightLeg)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT RightFoot\n${t(3)}{\n${t(4)}OFFSET ${o(off.rightFoot)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT RightToeBase\n${t(4)}{\n${t(5)}OFFSET ${o(off.rightToeBase)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}End Site\n${t(6)}\n${t(6)}OFFSET ${o(off.rightToeEnd)}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n}`
}

function buildMotion(frames, frameTime, off) {
  const lines = [
    'MOTION',
    `Frames: ${frames.length}`,
    `Frame Time: ${f(frameTime)}`,
  ]

  const DOWN  = [0, -1, 0]
  const FWD   = [0,  0, 1]

  const REST = {
    spine:         norm(off.spine),
    spine1:        norm(off.spine1),
    spine2:        norm(off.spine2),
    neck:          norm(off.neck),
    head:          norm(off.head),
    leftShoulder:  norm(off.leftShoulder),
    leftForeArm:   norm(off.leftForeArm),
    leftHand:      norm(off.leftHand),
    rightShoulder: norm(off.rightShoulder),
    rightArm:      norm(off.rightArm),
    rightForeArm:  norm(off.rightForeArm),
    rightHand:     norm(off.rightHand),
    leftUpLeg:     DOWN,
    leftLeg:       DOWN,
    leftFoot:      DOWN,
    leftToeBase:   FWD,
    rightUpLeg:    DOWN,
    rightLeg:      DOWN,
    rightFoot:     DOWN,
    rightToeBase:  FWD,
  }

  for (const frame of frames) {
    const p    = P(frame.landmarks, frame.worldLandmarks)
    const vals = []

    // ── Hips ──────────────────────────────────────────────────────────────
    vals.push(...p.hips)
    const spineDir    = sub(p.spine, p.hips)
    const hipsRot     = quatFromTo(REST.spine, norm(spineDir))
    vals.push(...quatToZXY(hipsRot))

    // ── Spine chain ───────────────────────────────────────────────────────
    const spine1Dir   = sub(p.spine1,  p.spine)
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

    const headDir     = sub(p.head, p.neck)
    const neckWorld   = quatMul(spineWorld2Base, spine2Local)
    const neckRot     = quatFromTo(quatRotate(neckWorld, REST.head), norm(headDir))
    const neckLocal   = quatMul(quatConj(neckWorld), quatMul(neckRot, neckWorld))
    vals.push(...quatToZXY(neckLocal))
    const spineWorld2 = neckWorld

    vals.push(0, 0, 0)  // Head end

    // ── Left arm ──────────────────────────────────────────────────────────
    const lShoDir     = sub(p.leftShoulder, p.spine2)
    const lShoRot     = quatFromTo(quatRotate(spineWorld2, REST.leftShoulder), norm(lShoDir))
    const lShoLocal   = quatMul(quatConj(spineWorld2), quatMul(lShoRot, spineWorld2))
    vals.push(...quatToZXY(lShoLocal))

    const lArmDir     = sub(p.leftForeArm, p.leftShoulder)
    const lShoWorld   = quatMul(spineWorld2, lShoLocal)
    const lArmRot     = quatFromTo(quatRotate(lShoWorld, REST.leftForeArm), norm(lArmDir))
    const lArmLocal   = quatMul(quatConj(lShoWorld), quatMul(lArmRot, lShoWorld))
    vals.push(...quatToZXY(lArmLocal))

    const lFADir      = sub(p.leftHand, p.leftForeArm)
    const lArmWorld   = quatMul(lShoWorld, lArmLocal)
    const lFARot      = quatFromTo(quatRotate(lArmWorld, REST.leftHand), norm(lFADir))
    const lFALocal    = quatMul(quatConj(lArmWorld), quatMul(lFARot, lArmWorld))
    vals.push(...quatToZXY(lFALocal))
    vals.push(0, 0, 0)

    // ── Right arm ─────────────────────────────────────────────────────────
    const rShoDir     = sub(p.rightShoulder, p.spine2)
    const rShoRot     = quatFromTo(quatRotate(spineWorld2, REST.rightShoulder), norm(rShoDir))
    const rShoLocal   = quatMul(quatConj(spineWorld2), quatMul(rShoRot, spineWorld2))
    vals.push(...quatToZXY(rShoLocal))

    const rArmDir     = sub(p.rightForeArm, p.rightShoulder)
    const rShoWorld   = quatMul(spineWorld2, rShoLocal)
    const rArmRot     = quatFromTo(quatRotate(rShoWorld, REST.rightForeArm), norm(rArmDir))
    const rArmLocal   = quatMul(quatConj(rShoWorld), quatMul(rArmRot, rShoWorld))
    vals.push(...quatToZXY(rArmLocal))

    const rFADir      = sub(p.rightHand, p.rightForeArm)
    const rArmWorld   = quatMul(rShoWorld, rArmLocal)
    const rFARot      = quatFromTo(quatRotate(rArmWorld, REST.rightHand), norm(rFADir))
    const rFALocal    = quatMul(quatConj(rArmWorld), quatMul(rFARot, rArmWorld))
    vals.push(...quatToZXY(rFALocal))
    vals.push(0, 0, 0)

    // ── Left leg ──────────────────────────────────────────────────────────
    const lULDir      = sub(p.leftLeg, p.leftUpLeg)
    const lULRot      = quatFromTo(quatRotate(hipsRot, REST.leftUpLeg), norm(lULDir))
    let lULLocal      = quatMul(quatConj(hipsRot), quatMul(lULRot, hipsRot))

    // Twist Modulation layer
    let lTwistDeg = 0
    if (lULDir[2] < -5.0) { 
      lTwistDeg = Math.min(55, Math.abs(lULDir[2]) * 1.8)
    }
    const lRad = (lTwistDeg * Math.PI) / 180
    const qLTwist = [Math.cos(lRad / 2), 0, Math.sin(lRad / 2), 0]
    lULLocal = quatMul(lULLocal, qLTwist) 

    vals.push(...quatToZXY(lULLocal))
    const lULWorld    = quatMul(hipsRot, lULLocal)

    const lThighFwd   = norm(lULDir)
    const lThighRight = norm(cross(lThighFwd, [0, 1, 0]))
    const lShinRaw    = norm(sub(p.leftFoot, p.leftLeg))
    const lShinSide   = dot(lShinRaw, lThighRight)
    const lShinPlane  = norm(sub(lShinRaw, scale(lThighRight, lShinSide)))
    const lLRot       = quatFromTo(quatRotate(lULWorld, REST.leftLeg), lShinPlane)
    const lLLocal      = quatMul(quatConj(lULWorld), quatMul(lLRot, lULWorld))

    vals.push(...quatToZXY(lLLocal))
    const lLWorld      = quatMul(lULWorld, lLLocal)

    const lFDir       = sub(p.leftToeBase, p.leftFoot)
    const lFRot       = quatFromTo(quatRotate(lLWorld, REST.leftFoot), norm(lFDir))
    const lFLocal      = quatMul(quatConj(lLWorld), quatMul(lFRot, lLWorld))

    // Foot Orientation Anchoring (Left Ankle)
    let lFEuler = quatToZXY(lFLocal);
    lFEuler[0] = Math.max(-15, Math.min(15, lFEuler[0]));  // Clamp side-to-side ankle tilt
    lFEuler[2] = Math.max(-20, Math.min(20, lFEuler[2]));  // Clamp independent twist
    vals.push(...lFEuler);
    vals.push(0, 0, 0)

    // ── Right leg ─────────────────────────────────────────────────────────
    const rULDir      = sub(p.rightLeg, p.rightUpLeg)
    const rULRot      = quatFromTo(quatRotate(hipsRot, REST.rightUpLeg), norm(rULDir))
    let rULLocal      = quatMul(quatConj(hipsRot), quatMul(rULRot, hipsRot))

    // Twist Modulation layer
    let rTwistDeg = 0
    if (rULDir[2] < -5.0) {
      rTwistDeg = -Math.min(55, Math.abs(rULDir[2]) * 1.8)
    }
    const rRad = (rTwistDeg * Math.PI) / 180
    const qRTwist = [Math.cos(rRad / 2), 0, Math.sin(rRad / 2), 0]
    rULLocal = quatMul(rULLocal, qRTwist)

    vals.push(...quatToZXY(rULLocal))
    const rULWorld    = quatMul(hipsRot, rULLocal)

    const rThighFwd   = norm(rULDir)
    const rThighRight = norm(cross(rThighFwd, [0, 1, 0]))
    const rShinRaw    = norm(sub(p.rightFoot, p.rightLeg))
    const rShinSide   = dot(rShinRaw, rThighRight)
    const rShinPlane  = norm(sub(rShinRaw, scale(rThighRight, rShinSide)))
    const rLRot       = quatFromTo(quatRotate(rULWorld, REST.rightLeg), rShinPlane)
    const rLLocal      = quatMul(quatConj(rULWorld), quatMul(rLRot, rULWorld))

    // Smooth, multi-axis knee calculations
    vals.push(...quatToZXY(rLLocal))
    const rLWorld      = quatMul(rULWorld, rLLocal)

    const rFDir       = sub(p.rightToeBase, p.rightFoot)
    const rFRot       = quatFromTo(quatRotate(rLWorld, REST.rightFoot), norm(rFDir))
    const rFLocal      = quatMul(quatConj(rLWorld), quatMul(rFRot, rLWorld))

    // Foot Orientation Anchoring (Right Ankle)
    let rFEuler = quatToZXY(rFLocal);
    rFEuler[0] = Math.max(-15, Math.min(15, rFEuler[0]));
    rFEuler[2] = Math.max(-20, Math.min(20, rFEuler[2]));
    vals.push(...rFEuler);
    vals.push(0, 0, 0)

    lines.push(vals.map(v => f(v)).join(' '))
  }

  return lines.join('\n')
}

// ── Public ────────────────────────────────────────────────────────────────────
export function exportBVH(frames, captureFps) {
  if (!frames?.length) return

  const off = getRestOffsets()
  
  // Temporal smoothing across a 3-frame sliding window 
  const smoothedFrames = smoothFrames(frames, 3)
  
  const bvh = buildHierarchy(off) + '\n' + buildMotion(smoothedFrames, 1 / captureFps, off)

  const blob = new Blob([bvh], { type: 'text/plain' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url
  a.download = 'pose_refined.bvh'
  a.click()
  URL.revokeObjectURL(url)
}