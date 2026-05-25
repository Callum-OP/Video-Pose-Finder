import { solveIK } from './ikSolver.js'


// Joint structure verified to load in Clip Studio Paint.
// Matches Mixamo/Blender BVH export convention exactly.
//
// 22 joints with CHANNELS (non-end-sites):
//   Hips = 6 channels (Xpos Ypos Zpos Zrot Xrot Yrot)
//   All others = 3 channels (Zrot Xrot Yrot)
//   Total per frame = 6 + 21*3 = 69 values
//
// End Sites (no channels, just OFFSET):
//   Head > End Site
//   LeftHand > End Site
//   RightHand > End Site
//   LeftToeBase > End Site
//   RightToeBase > End Site

// ── Coordinate conversion ─────────────────────────────────────────────────────
// MediaPipe provides two landmark sets:
//   landmarks        — normalised image XY (0→1), estimated Z depth (unreliable)
//   worldLandmarks   — real metric 3D (metres, hip-centred), Y down, Z toward camera
//
// We use worldLandmarks for BVH positions since they have real 3D structure.
// Transform to BVH space: flip Y (down→up) and flip Z (toward-cam → into-screen)
// X is already correct — MediaPipe world X matches character X (right = positive)
// when facing camera (landmark 11 = character's left shoulder = negative X in world)
const SCALE = 100  // metres → cm-ish units CSP expects

function mp(lms, worldLms, idx) {
  const src = worldLms?.[idx] ?? lms[idx]
  if (!src) return [0, 0, 0]
  if (worldLms?.[idx]) {
    // MediaPipe world space: X is mirrored relative to character space.
    // Person faces camera → their left shoulder (mp11) has positive X in world,
    // but BVH LeftShoulder should have negative X (character's left = -X in BVH).
    // Flip X and Y, keep Z positive (toward camera = forward for character).
    return [-src.x * SCALE, -src.y * SCALE, src.z * SCALE]
  }
  return [-src.x * SCALE, -src.y * SCALE, 0]
}

function avg(a, b) {
  return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2]
}

function sub(a, b) {
  return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

function scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s]
}

function norm(v) {
  const l = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) || 1e-8
  return [v[0]/l, v[1]/l, v[2]/l]
}

function len(v) {
  return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
}

// Named positions from MediaPipe landmarks
function P(lms, worldLms) {
  const w = worldLms  // shorthand
  const leftHip   = mp(lms, w, 23)
  const rightHip  = mp(lms, w, 24)
  const leftSho   = mp(lms, w, 11)
  const rightSho  = mp(lms, w, 12)
  const hips      = avg(leftHip, rightHip)
  const shoulders = avg(leftSho, rightSho)
  const spine     = avg(hips, shoulders)
  const spine1    = [(hips[0]+shoulders[0]*2)/3, (hips[1]+shoulders[1]*2)/3, (hips[2]+shoulders[2]*2)/3]
  const spine2    = shoulders

  // Neck top = midpoint of ears (landmarks 7=left ear, 8=right ear)
  // This gives a much better "top of neck" than the nose which hunches forward
  const leftEar  = mp(lms, w, 7)
  const rightEar = mp(lms, w, 8)
  const earMid   = avg(leftEar, rightEar)
  const nose     = mp(lms, w, 0)

  return {
    hips,
    spine,
    spine1,
    spine2,
    neck:          avg(shoulders, earMid),  // halfway between shoulders and ear midpoint
    head:          earMid,                  // ear midpoint as head position
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
// BVH rotations are LOCAL — each joint's rotation is relative to its parent.
// Strategy: for each bone, compute the quaternion that rotates the T-pose
// bone direction to the current bone direction, then extract ZXY Euler angles.

function cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ]
}

function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// Quaternion that rotates unit vector 'from' to unit vector 'to'
function quatFromTo(from, to) {
  const f = norm(from)
  const t = norm(to)
  const d = dot(f, t)

  if (d >= 1.0 - 1e-6) return [1, 0, 0, 0]  // no rotation needed
  if (d <= -1.0 + 1e-6) {
    // 180° rotation — find any perpendicular axis
    let perp = cross(f, [1,0,0])
    if (dot(perp,perp) < 1e-6) perp = cross(f, [0,1,0])
    const ax = norm(perp)
    return [0, ax[0], ax[1], ax[2]]  // 180° = w=0
  }
  const axis = cross(f, t)
  const w    = Math.sqrt((1+d) / 2)
  const s    = 1 / (2 * w)
  return [w, axis[0]*s, axis[1]*s, axis[2]*s]
}

// Multiply two quaternions
function quatMul([w1,x1,y1,z1], [w2,x2,y2,z2]) {
  return [
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
  ]
}

// Conjugate (inverse for unit quaternion)
function quatConj([w,x,y,z]) { return [w,-x,-y,-z] }

// Rotate a vector by a quaternion
function quatRotate([w,x,y,z], [vx,vy,vz]) {
  const q = [w,x,y,z]
  const qv = [0, vx, vy, vz]
  const [,rx,ry,rz] = quatMul(quatMul(q, qv), quatConj(q))
  return [rx, ry, rz]
}

// Extract ZXY Euler angles (degrees) from a quaternion
// CSP BVH channel order: Zrotation Xrotation Yrotation
function quatToZXY([w,x,y,z]) {
  // Rotation matrix elements needed for ZXY decomposition
  const m = [
    1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
    2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
    2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
  ]
  // ZXY: Rx = asin(m[7]), Ry = atan2(-m[6], m[8]), Rz = atan2(-m[1], m[4])
  const rx = Math.asin(Math.max(-1, Math.min(1, m[7])))
  const ry = Math.atan2(-m[6], m[8])
  const rz = Math.atan2(-m[1], m[4])
  const deg = r => r * (180/Math.PI)
  return [deg(rz), deg(rx), deg(ry)]  // Zrot, Xrot, Yrot
}

// For a given bone: compute local rotation quaternion.
// restDir = T-pose bone direction (from offsets)
// currentDir = current bone direction (from live landmarks)
// parentWorldQuat = accumulated world rotation of parent joint
function localBoneRot(restDir, currentDir, parentWorldQuat) {
  // What direction does the rest bone point in world space given parent rotation?
  const restWorld = parentWorldQuat ? quatRotate(parentWorldQuat, norm(restDir)) : norm(restDir)
  // Rotation from rest world direction to current world direction
  const worldRot = quatFromTo(restWorld, norm(currentDir))
  // Convert to local: undo parent rotation
  return parentWorldQuat ? quatMul(quatConj(parentWorldQuat), quatMul(worldRot, parentWorldQuat)) : worldRot
}

// ── Hardcoded T-pose rest offsets ─────────────────────────────────────────────
// Using averaged frame offsets caused compounded rotation errors because the
// video is never actually in a T-pose. These are anatomical proportions that
// match what CSP's drawing figure expects — a standard humanoid T-pose.
// Units are in the same SCALE space (100 = ~1m).

function getRestOffsets() {
  return {
    hips:          [0,      0,     0   ],
    spine:         [0,      10,    0   ],
    spine1:        [0,      10,    0   ],
    spine2:        [0,      10,    0   ],
    neck:          [0,      8,     0   ],
    head:          [0,      8,     0   ],
    headEnd:       [0,      8,     0   ],
    // Arms point straight out to sides in T-pose
    leftShoulder:  [-8,     0,     0   ],
    leftArm:       [0,      0,     0   ],  // coincident with shoulder
    leftForeArm:   [-15,    0,     0   ],
    leftHand:      [-12,    0,     0   ],
    leftHandEnd:   [-12,    0,     0   ],
    rightShoulder: [8,      0,     0   ],
    rightArm:      [0,      0,     0   ],
    rightForeArm:  [15,     0,     0   ],
    rightHand:     [12,     0,     0   ],
    rightHandEnd:  [12,     0,     0   ],
    // Legs point straight down in T-pose — X offset is the hip position,
    // not the bone direction. Diagonal rest dirs cause sideways knee rotation.
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

// ── Format helpers ────────────────────────────────────────────────────────────
const f = n => n.toFixed(4)
const o = ([x,y,z]) => `${f(x)} ${f(y)} ${f(z)}`
const t = n => '\t'.repeat(n)

// ── HIERARCHY ─────────────────────────────────────────────────────────────────
function buildHierarchy(off) {
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
${t(8)}End Site
${t(8)}{
${t(9)}OFFSET ${o(off.leftHandEnd)}
${t(8)}}
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
${t(8)}End Site
${t(8)}{
${t(9)}OFFSET ${o(off.rightHandEnd)}
${t(8)}}
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

// ── MOTION ────────────────────────────────────────────────────────────────────
// Channel order must exactly match HIERARCHY declaration order:
// Hips: Xpos Ypos Zpos Zrot Xrot Yrot (6)
// Then for each joint in declaration order: Zrot Xrot Yrot (3 each)
// Joints in order: Spine Spine1 Spine2 Neck Head
//                  LeftShoulder LeftArm LeftForeArm LeftHand
//                  RightShoulder RightArm RightForeArm RightHand
//                  LeftUpLeg LeftLeg LeftFoot LeftToeBase
//                  RightUpLeg RightLeg RightFoot RightToeBase
// = 22 joints: 6 + 21*3 = 69 values per frame

function buildMotion(frames, frameTime, off) {
  const lines = [
    'MOTION',
    `Frames: ${frames.length}`,
    `Frame Time: ${f(frameTime)}`,
  ]

  const DOWN  = [0, -1, 0]
  const UP    = [0,  1, 0]
  const FWD   = [0,  0, 1]

  // Rest directions: normalised bone directions in T-pose.
  // Legs point straight DOWN — the X offset in the hierarchy offset is the
  // hip socket position, NOT the bone direction. Using the diagonal would
  // cause the rotation solver to introduce spurious sideways knee rotation.
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
    rightForeArm:  norm(off.rightForeArm),
    rightHand:     norm(off.rightHand),
    leftUpLeg:     DOWN,   // straight down, not diagonal
    leftLeg:       DOWN,
    leftFoot:      DOWN,
    leftToeBase:   FWD,
    rightUpLeg:    DOWN,   // straight down, not diagonal
    rightLeg:      DOWN,
    rightFoot:     DOWN,
    rightToeBase:  FWD,
  }

  const IDENTITY = [1,0,0,0]
  const deg = r => r * (180/Math.PI)

  for (const frame of frames) {
    const p    = P(frame.landmarks, frame.worldLandmarks)
    const vals = []

    // ── Hips: world translation + world rotation ──────────────────────────
    vals.push(...p.hips)

    // Hips rotation: from rest up direction to actual spine direction
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

    // Spine2: add torso twist from shoulder-to-shoulder vector.
    // The shoulder line tells us how much the chest is rotated around Y (twist).
    // We extract this by comparing the actual shoulder vector to what the
    // spine orientation predicts the shoulder line should be.
    const spineWorld2Base = quatMul(spineWorld1, spine1Local)
    const actualShoVec    = norm(sub(p.rightShoulder, p.leftShoulder))
    // Expected shoulder direction in current spine orientation (spine2 rest = sideways)
    const expectedShoVec  = quatRotate(spineWorld2Base, norm(off.rightShoulder))
    // Twist = rotation from expected to actual shoulder line, projected onto spine axis
    const spineAxis       = norm(spine2Dir.length !== undefined ? spine2Dir : sub(p.spine2, p.spine1))
    const twistQuat       = quatFromTo(
      norm([expectedShoVec[0], 0, expectedShoVec[2]]),  // horizontal component only
      norm([actualShoVec[0],   0, actualShoVec[2]])
    )
    // Combine spine2 direction rotation with twist
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
    const spineWorld2 = neckWorld  // alias for arm code below

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
    // Hip: full 3D rotation to aim thigh at knee
    const lULDir      = sub(p.leftLeg, p.leftUpLeg)
    const lULRot      = quatFromTo(quatRotate(hipsRot, REST.leftUpLeg), norm(lULDir))
    const lULLocal    = quatMul(quatConj(hipsRot), quatMul(lULRot, hipsRot))
    vals.push(...quatToZXY(lULLocal))
    const lULWorld    = quatMul(hipsRot, lULLocal)

    // Knee: constrain to sagittal plane of the thigh.
    // The thigh defines a local coordinate frame. The knee should only bend
    // on the axis perpendicular to both the thigh direction and the sideways axis.
    // We project the shin vector onto the thigh's sagittal plane (removing sideways)
    // then compute the rotation only within that plane.
    const lThighFwd   = norm(lULDir)
    const lThighRight = norm(cross(lThighFwd, [0, 1, 0]))  // thigh's local right axis
    const lShinRaw    = norm(sub(p.leftFoot, p.leftLeg))
    // Remove sideways component from shin direction
    const lShinSide   = dot(lShinRaw, lThighRight)
    const lShinPlane  = norm(sub(lShinRaw, scale(lThighRight, lShinSide)))
    const lLRot       = quatFromTo(quatRotate(lULWorld, REST.leftLeg), lShinPlane)
    const lLLocal     = quatMul(quatConj(lULWorld), quatMul(lLRot, lULWorld))
    vals.push(...quatToZXY(lLLocal))

    const lFDir       = sub(p.leftToeBase, p.leftFoot)
    const lLWorld     = quatMul(lULWorld, lLLocal)
    const lFRot       = quatFromTo(quatRotate(lLWorld, REST.leftFoot), norm(lFDir))
    const lFLocal     = quatMul(quatConj(lLWorld), quatMul(lFRot, lLWorld))
    vals.push(...quatToZXY(lFLocal))
    vals.push(0, 0, 0)

    // ── Right leg ─────────────────────────────────────────────────────────
    const rULDir      = sub(p.rightLeg, p.rightUpLeg)
    const rULRot      = quatFromTo(quatRotate(hipsRot, REST.rightUpLeg), norm(rULDir))
    const rULLocal    = quatMul(quatConj(hipsRot), quatMul(rULRot, hipsRot))
    vals.push(...quatToZXY(rULLocal))
    const rULWorld    = quatMul(hipsRot, rULLocal)

    const rThighFwd   = norm(rULDir)
    const rThighRight = norm(cross(rThighFwd, [0, 1, 0]))
    const rShinRaw    = norm(sub(p.rightFoot, p.rightLeg))
    const rShinSide   = dot(rShinRaw, rThighRight)
    const rShinPlane  = norm(sub(rShinRaw, scale(rThighRight, rShinSide)))
    const rLRot       = quatFromTo(quatRotate(rULWorld, REST.rightLeg), rShinPlane)
    const rLLocal     = quatMul(quatConj(rULWorld), quatMul(rLRot, rULWorld))
    vals.push(...quatToZXY(rLLocal))

    const rFDir       = sub(p.rightToeBase, p.rightFoot)
    const rLWorld     = quatMul(rULWorld, rLLocal)
    const rFRot       = quatFromTo(quatRotate(rLWorld, REST.rightFoot), norm(rFDir))
    const rFLocal     = quatMul(quatConj(rLWorld), quatMul(rFRot, rLWorld))
    vals.push(...quatToZXY(rFLocal))
    vals.push(0, 0, 0)

    lines.push(vals.map(v => f(v)).join(' '))
  }

  return lines.join('\n')
}

// ── Public ────────────────────────────────────────────────────────────────────
export function exportBVH(frames, captureFps) {
  if (!frames?.length) return

  // IK solver disabled — raw world landmarks give closer results than
  // FABRIK with hardcoded pole vectors that don't adapt to camera angle

  const off = getRestOffsets()
  const bvh = buildHierarchy(off) + '\n' + buildMotion(frames, 1 / captureFps, off)

  const blob = new Blob([bvh], { type: 'text/plain' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url
  a.download = 'pose_raw.bvh'
  a.click()
  URL.revokeObjectURL(url)
}