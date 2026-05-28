// ── BVH Exporter ─────────────────────────────────────────────────────────────
// Converts MediaPipe pose landmark frames to a BVH animation file.
// Matches Mixamo/Blender BVH export convention exactly.

const SCALE = 100

function mp(lms, worldLms, idx) {
  const src = worldLms?.[idx] ?? lms[idx]
  if (!src) return [0, 0, 0]
  if (worldLms?.[idx]) return [-src.x * SCALE, -src.y * SCALE, src.z * SCALE]
  return [-src.x * SCALE, -src.y * SCALE, 0]
}

function avg(a, b) { return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2] }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
function scale(v, s) { return [v[0]*s, v[1]*s, v[2]*s] }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
function len(v) { return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) }
function norm(v) { const l = len(v) || 1e-8; return [v[0]/l, v[1]/l, v[2]/l] }
function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

// Rodrigues rotation
function rotateAround(vec, ax, rad) {
  const c = Math.cos(rad), s = Math.sin(rad)
  return add(add(scale(vec, c), scale(cross(ax, vec), s)), scale(ax, dot(ax, vec) * (1-c)))
}

// ── Place knee correctly ───────────────────────────────
function constrainKnee(hip, rawKnee, ankle, heel, toe, boneLenThigh, boneLenShin, pelvisFwd) {
  // Determine the hinge axis
  // Use the raw thigh direction as a starting approximation.
  const rawThighDir = norm(sub(rawKnee, hip))

  // Heel and toe gives the foot's facing direction
  const footVec = sub(toe, heel)
  let hingeAxis
  if (len(footVec) > 0.001) {
    // Project foot vector onto the plane perpendicular to the thigh
    const footAlongThigh = dot(footVec, rawThighDir)
    const footPerp = norm(sub(footVec, scale(rawThighDir, footAlongThigh)))
    if (len(footPerp) > 0.001) {
      // Hinge axis = cross(thigh, footForward-perpendicular)
      // This is the axis the kneecap rotates around — perpendicular to both
      hingeAxis = norm(cross(rawThighDir, footPerp))
    }
  }
  // Fallback: knee bends in the world sagittal plane (forward/back)
  if (!hingeAxis) hingeAxis = norm(cross(rawThighDir, [0, 0, 1]))

  // Solve the knee position as a 2-bone IK in the hinge plane

  const hipToAnkle    = sub(ankle, hip)
  const dist          = len(hipToAnkle)
  const totalLen      = boneLenThigh + boneLenShin

  // If ankle is unreachable, fully extend in the hinge plane toward ankle
  if (dist >= totalLen) {
    // Project ankle direction onto hinge plane
    const toAnkleDir = norm(hipToAnkle)
    const alongHinge = dot(toAnkleDir, hingeAxis)
    const inPlane    = norm(sub(toAnkleDir, scale(hingeAxis, alongHinge)))
    return add(hip, scale(inPlane, boneLenThigh))
  }

  // Find angle at hip between thigh and hip to ankle
  const d = Math.max(dist, 0.001)
  const cosHipAngle = Math.max(-1, Math.min(1,
    (boneLenThigh**2 + d**2 - boneLenShin**2) / (2 * boneLenThigh * d)
  ))
  const hipAngle = Math.acos(cosHipAngle)

  // Clamp knee bend angle (0° = straight, 150° = max bend)
  const cosKneeAngle = Math.max(-1, Math.min(1,
    (boneLenThigh**2 + boneLenShin**2 - d**2) / (2 * boneLenThigh * boneLenShin)
  ))
  const bendDeg = 180 - Math.acos(cosKneeAngle) * 180 / Math.PI
  let effectiveHipAngle = hipAngle
  if (bendDeg > 150) {
    // Recompute hip angle for max bend
    const clampedKneeAngle = 30 * Math.PI / 180  // 150° bend = 30° interior angle
    const sinH = Math.sin(clampedKneeAngle) * boneLenShin / d
    effectiveHipAngle = Math.asin(Math.max(-1, Math.min(1, sinH)))
  }

  // Place knee in the hinge plane
  const ankleAlongH  = dot(hipToAnkle, hingeAxis)
  const hipToAnkleIP = sub(hipToAnkle, scale(hingeAxis, ankleAlongH))
  const planeFwd     = len(hipToAnkleIP) > 0.001
    ? norm(hipToAnkleIP)
    : norm(cross(hingeAxis, [0, 1, 0]))

  // The direction the knee bends toward, perpendicular to planeFwd
  let planeSide = norm(cross(hingeAxis, planeFwd))

  // The knee always bends so the shin goes BEHIND the thigh, never in front.
  const rawKneeOffset = sub(rawKnee, hip)
  const rawKneeAlongH = dot(rawKneeOffset, hingeAxis)
  const rawKneeInPlane = norm(sub(rawKneeOffset, scale(hingeAxis, rawKneeAlongH)))
  // Get which side of planeFwd the knee is on
  const rawSideComponent = dot(rawKneeInPlane, planeSide)
  if (rawSideComponent < 0) planeSide = scale(planeSide, -1)

  // Knee must bend backward relative to the body, not forward.
  if (dot(planeSide, pelvisFwd) > 0) planeSide = scale(planeSide, -1)

  // Knee direction = rotate planeFwd by hipAngle toward planeSide
  const kneeDir = add(
    scale(planeFwd, Math.cos(effectiveHipAngle)),
    scale(planeSide, Math.sin(effectiveHipAngle))
  )

  return add(hip, scale(norm(kneeDir), boneLenThigh))
}

// ── Elbow: use raw MediaPipe position ────────────────────────
function constrainElbow(shoulder, rawElbow, wrist, boneLenUpper, boneLenFore) {
  // Enforce upper arm length from shoulder
  const upperDir    = norm(sub(rawElbow, shoulder))
  const fixedElbow  = add(shoulder, scale(upperDir, boneLenUpper))
  return fixedElbow
}

// ── Extract and constrain all joint positions from a frame ───────────────────
function P(lms, worldLms) {
  const w = worldLms

  const leftHip    = mp(lms, w, 23)
  const rightHip   = mp(lms, w, 24)
  const leftSho    = mp(lms, w, 11)
  const rightSho   = mp(lms, w, 12)
  const hips       = avg(leftHip, rightHip)
  const shoulders  = avg(leftSho, rightSho)
  const spine      = avg(hips, shoulders)
  const spine1     = [(hips[0]+shoulders[0]*2)/3, (hips[1]+shoulders[1]*2)/3, (hips[2]+shoulders[2]*2)/3]
  const spine2     = shoulders
  const leftEar    = mp(lms, w, 7)
  const rightEar   = mp(lms, w, 8)
  const earMid     = avg(leftEar, rightEar)

  // Raw ankle and wrist positions, these are trusted and never moved
  const lAnkle = mp(lms, w, 27)
  const rAnkle = mp(lms, w, 28)
  const lWrist = mp(lms, w, 15)
  const rWrist = mp(lms, w, 16)

  // Foot landmarks for knee hinge axis
  const lHeel  = mp(lms, w, 29)
  const rHeel  = mp(lms, w, 30)
  const lToe   = mp(lms, w, 31)
  const rToe   = mp(lms, w, 32)

  // Raw knee and elbow, used for initial direction estimate
  const rawLKnee  = mp(lms, w, 25)
  const rawRKnee  = mp(lms, w, 26)
  const rawLElbow = mp(lms, w, 13)
  const rawRElbow = mp(lms, w, 14)

  // Measure bone lengths from the raw positions
  const lThighLen = len(sub(rawLKnee,  leftHip))   || 1
  const rThighLen = len(sub(rawRKnee,  rightHip))  || 1
  const lShinLen  = len(sub(lAnkle,    rawLKnee))  || 1
  const rShinLen  = len(sub(rAnkle,    rawRKnee))  || 1
  const lUpperLen = len(sub(rawLElbow, leftSho))   || 1
  const rUpperLen = len(sub(rawRElbow, rightSho))  || 1
  const lForeLen  = len(sub(lWrist,    rawLElbow)) || 1
  const rForeLen  = len(sub(lWrist,    rawRElbow)) || 1

  // ── Pelvis forward axis ──────────────────────────────────────────────────
  // Derived from landmark geometry so it stays correct at any body angle.
  const hipRight  = norm(sub(rightHip, leftHip)) // Left hip to right hip
  const spineUp   = norm(sub(avg(leftSho, rightSho), avg(leftHip, rightHip))) // Hip midpoint to shoulder midpoint
  // Project spineUp perpendicular to hipRight. Giving a more accurate forward direction for the pelvis, especially when the person is leaning or turned.
  const spineUpOrtho = norm(sub(spineUp, scale(hipRight, dot(spineUp, hipRight))))
  const pelvisFwd = norm(cross(hipRight, spineUpOrtho)) // Direction the pelvis faces

  // Knee hinge constraint
  const leftKnee  = constrainKnee(leftHip,  rawLKnee, lAnkle, lHeel, lToe, lThighLen, lShinLen, pelvisFwd)
  const rightKnee = constrainKnee(rightHip, rawRKnee, rAnkle, rHeel, rToe, rThighLen, rShinLen, pelvisFwd)

  // Elbow, enforce bone length consistency
  const leftElbow  = constrainElbow(leftSho,  rawLElbow, lWrist, lUpperLen, lForeLen)
  const rightElbow = constrainElbow(rightSho, rawRElbow, rWrist, rUpperLen, rForeLen)

  // ── Hand separation ──────────────────────────────────────────────────────
  // If the two wrists are closer than a minimum hand-width, nudge them apart
  const shoulderWidth  = len(sub(rightSho, leftSho))
  const minHandSep     = shoulderWidth * 0.18
  const wristVec       = sub(rWrist, lWrist)
  const wristDist      = len(wristVec)
  let adjLWrist = lWrist, adjRWrist = rWrist
  if (wristDist < minHandSep && wristDist > 0.001) {
    // Nudge both wrists apart along the shoulder axis by equal amounts
    const deficit   = (minHandSep - wristDist) / 2
    const pushDir   = norm(sub(rightSho, leftSho))
    adjLWrist = sub(lWrist, scale(pushDir, deficit))
    adjRWrist = add(rWrist, scale(pushDir, deficit))
  }

  // ── Foot roll limits derived from heel and toe direction ─────────────────────
  const lFootVec     = sub(lToe, lHeel)
  const lShinDir     = norm(sub(lAnkle, leftKnee))
  const lFootLateral = len(lFootVec) > 0.001
    ? Math.abs(dot(norm(lFootVec), norm(cross(lShinDir, spineUpOrtho)))) // lateral component
    : 0
  const lFootRollLimit  = Math.max(20, Math.min(40, 20 + lFootLateral * 40))  // 20°–40°
  const lFootTwistLimit = Math.max(25, Math.min(50, 25 + lFootLateral * 50))  // 25°–50°

  const rFootVec     = sub(rToe, rHeel)
  const rShinDir     = norm(sub(rAnkle, rightKnee))
  const rFootLateral = len(rFootVec) > 0.001
    ? Math.abs(dot(norm(rFootVec), norm(cross(rShinDir, spineUpOrtho))))
    : 0
  const rFootRollLimit  = Math.max(20, Math.min(40, 20 + rFootLateral * 40))
  const rFootTwistLimit = Math.max(25, Math.min(50, 25 + rFootLateral * 50))

  return {
    hips, spine, spine1, spine2,
    neck:          avg(shoulders, earMid),
    head:          earMid,
    leftShoulder:  leftSho,
    leftArm:       leftSho,
    leftForeArm:   leftElbow,
    leftHand:      adjLWrist,
    rightShoulder: rightSho,
    rightArm:      rightSho,
    rightForeArm:  rightElbow,
    rightHand:     adjRWrist,
    leftUpLeg:     leftHip,
    leftLeg:       leftKnee,
    leftFoot:      lAnkle,
    leftToeBase:   mp(lms, w, 31),
    rightUpLeg:    rightHip,
    rightLeg:      rightKnee,
    rightFoot:     rAnkle,
    rightToeBase:  mp(lms, w, 32),
    // Roll limits per foot, (limit foot twist/roll more if foot is turned out more)
    lFootRollLimit, lFootTwistLimit,
    rFootRollLimit, rFootTwistLimit,
  }
}

// ── Rotation maths ────────────────────────────────────────────────────────────
function quatFromTo(from, to) {
  const f = norm(from), t = norm(to)
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

function smoothFrames(frames, windowSize = 3) {
  if (!frames || frames.length === 0) return []
  return frames.map((frame, i) => {
    const start = Math.max(0, i - Math.floor(windowSize / 2))
    const end   = Math.min(frames.length, start + windowSize)
    const windowFrames = frames.slice(start, end)
    if (!frame.worldLandmarks) return frame
    const isArray = Array.isArray(frame.worldLandmarks)
    const smoothed = isArray ? [] : {}
    for (const key of Object.keys(frame.worldLandmarks)) {
      let sx = 0, sy = 0, sz = 0, count = 0
      for (const wf of windowFrames) {
        const pt = wf.worldLandmarks?.[key]
        if (pt) { sx += pt.x; sy += pt.y; sz += pt.z; count++ }
      }
      if (count > 0) {
        const sp = { x: sx/count, y: sy/count, z: sz/count }
        if (isArray) smoothed[parseInt(key, 10)] = sp
        else smoothed[key] = sp
      }
    }
    return { ...frame, worldLandmarks: smoothed }
  })
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
  const q = [w,x,y,z], qv = [0,vx,vy,vz]
  const [,rx,ry,rz] = quatMul(quatMul(q, qv), quatConj(q))
  return [rx, ry, rz]
}
function quatToZXY([w,x,y,z]) {
  const m = [
    1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
    2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
    2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y),
  ]
  const rx = Math.asin(Math.max(-1, Math.min(1, m[7])))
  const ry = Math.atan2(-m[6], m[8])
  const rz = Math.atan2(-m[1], m[4])
  const deg = r => r * (180/Math.PI)
  return [deg(rz), deg(rx), deg(ry)]
}

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
  }
}

const f = n => n.toFixed(4)
const o = ([x,y,z]) => `${f(x)} ${f(y)} ${f(z)}`
const t = n => '\t'.repeat(n)

function buildHierarchy(off) {
  return `HIERARCHY\nROOT Hips\n{\n${t(1)}OFFSET ${o(off.hips)}\n${t(1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n${t(1)}JOINT Spine\n${t(1)}{\n${t(2)}OFFSET ${o(off.spine)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT Spine1\n${t(2)}{\n${t(3)}OFFSET ${o(off.spine1)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT Spine2\n${t(3)}{\n${t(4)}OFFSET ${o(off.spine2)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT Neck\n${t(4)}{\n${t(5)}OFFSET ${o(off.neck)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT Head\n${t(5)}{\n${t(6)}OFFSET ${o(off.head)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}End Site\n${t(6)}{\n${t(7)}OFFSET ${o(off.headEnd)}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(4)}JOINT LeftShoulder\n${t(4)}{\n${t(5)}OFFSET ${o(off.leftShoulder)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT LeftArm\n${t(5)}{\n${t(6)}OFFSET ${o(off.leftArm)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}JOINT LeftForeArm\n${t(6)}{\n${t(7)}OFFSET ${o(off.leftForeArm)}\n${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(7)}JOINT LeftHand\n${t(7)}{\n${t(8)}OFFSET ${o(off.leftHand)}\n${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(8)}End Site\n${t(8)}{\n${t(9)}OFFSET ${o(off.leftHandEnd)}\n${t(8)}}\n${t(7)}}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(4)}JOINT RightShoulder\n${t(4)}{\n${t(5)}OFFSET ${o(off.rightShoulder)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}JOINT RightArm\n${t(5)}{\n${t(6)}OFFSET ${o(off.rightArm)}\n${t(6)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(6)}JOINT RightForeArm\n${t(6)}{\n${t(7)}OFFSET ${o(off.rightForeArm)}\n${t(7)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(7)}JOINT RightHand\n${t(7)}{\n${t(8)}OFFSET ${o(off.rightHand)}\n${t(8)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(8)}End Site\n${t(8)}{\n${t(9)}OFFSET ${o(off.rightHandEnd)}\n${t(8)}}\n${t(7)}}\n${t(6)}}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n${t(1)}JOINT LeftUpLeg\n${t(1)}{\n${t(2)}OFFSET ${o(off.leftUpLeg)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT LeftLeg\n${t(2)}{\n${t(3)}OFFSET ${o(off.leftLeg)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT LeftFoot\n${t(3)}{\n${t(4)}OFFSET ${o(off.leftFoot)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT LeftToeBase\n${t(4)}{\n${t(5)}OFFSET ${o(off.leftToeBase)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}End Site\n${t(5)}{\n${t(6)}OFFSET ${o(off.leftToeEnd)}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n${t(1)}JOINT RightUpLeg\n${t(1)}{\n${t(2)}OFFSET ${o(off.rightUpLeg)}\n${t(2)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(2)}JOINT RightLeg\n${t(2)}{\n${t(3)}OFFSET ${o(off.rightLeg)}\n${t(3)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(3)}JOINT RightFoot\n${t(3)}{\n${t(4)}OFFSET ${o(off.rightFoot)}\n${t(4)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(4)}JOINT RightToeBase\n${t(4)}{\n${t(5)}OFFSET ${o(off.rightToeBase)}\n${t(5)}CHANNELS 3 Zrotation Xrotation Yrotation\n${t(5)}End Site\n${t(6)}\n${t(6)}OFFSET ${o(off.rightToeEnd)}\n${t(5)}}\n${t(4)}}\n${t(3)}}\n${t(2)}}\n${t(1)}}\n}`
}

function buildMotion(frames, frameTime, off) {
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

  for (const frame of frames) {
    // P() now applies hinge constraints before returning joint positions
    const p    = P(frame.landmarks, frame.worldLandmarks)
    const vals = []

    // ── Hips ──────────────────────────────────────────────────────────────
    vals.push(...p.hips)
    const spineDir   = sub(p.spine, p.hips)
    const hipsRot     = quatFromTo(REST.spine, norm(spineDir))
    vals.push(...quatToZXY(hipsRot))

    // ── Spine chain ───────────────────────────────────────────────────────
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

    // ── Left arm ──────────────────────────────────────────────────────────
    const lShoDir   = sub(p.leftShoulder, p.spine2)
    const lShoRot   = quatFromTo(quatRotate(spineWorld2, REST.leftShoulder), norm(lShoDir))
    const lShoLocal = quatMul(quatConj(spineWorld2), quatMul(lShoRot, spineWorld2))
    vals.push(...quatToZXY(lShoLocal))

    const lArmDir   = sub(p.leftForeArm, p.leftShoulder) // Constrained elbow
    const lShoWorld = quatMul(spineWorld2, lShoLocal)
    const lArmRot   = quatFromTo(quatRotate(lShoWorld, REST.leftForeArm), norm(lArmDir))
    const lArmLocal = quatMul(quatConj(lShoWorld), quatMul(lArmRot, lShoWorld))
    vals.push(...quatToZXY(lArmLocal))

    const lFADir    = sub(p.leftHand, p.leftForeArm)
    const lArmWorld = quatMul(lShoWorld, lArmLocal)
    const lFARot    = quatFromTo(quatRotate(lArmWorld, REST.leftHand), norm(lFADir))
    const lFALocal  = quatMul(quatConj(lArmWorld), quatMul(lFARot, lArmWorld))
    vals.push(...quatToZXY(lFALocal))
    vals.push(0, 0, 0)

    // ── Right arm ─────────────────────────────────────────────────────────
    const rShoDir   = sub(p.rightShoulder, p.spine2)
    const rShoRot   = quatFromTo(quatRotate(spineWorld2, REST.rightShoulder), norm(rShoDir))
    const rShoLocal = quatMul(quatConj(spineWorld2), quatMul(rShoRot, spineWorld2))
    vals.push(...quatToZXY(rShoLocal))

    const rArmDir   = sub(p.rightForeArm, p.rightShoulder) // Constrained elbow
    const rShoWorld = quatMul(spineWorld2, rShoLocal)
    const rArmRot   = quatFromTo(quatRotate(rShoWorld, REST.rightForeArm), norm(rArmDir))
    const rArmLocal = quatMul(quatConj(rShoWorld), quatMul(rArmRot, rShoWorld))
    vals.push(...quatToZXY(rArmLocal))

    const rFADir    = sub(p.rightHand, p.rightForeArm)
    const rArmWorld = quatMul(rShoWorld, rArmLocal)
    const rFARot    = quatFromTo(quatRotate(rArmWorld, REST.rightHand), norm(rFADir))
    const rFALocal  = quatMul(quatConj(rArmWorld), quatMul(rFARot, rArmWorld))
    vals.push(...quatToZXY(rFALocal))
    vals.push(0, 0, 0)

    // ── Left leg ──────────────────────────────────────────────────────────
    const lULDir   = sub(p.leftLeg, p.leftUpLeg) // Constrained knee
    const lULRot   = quatFromTo(quatRotate(hipsRot, REST.leftUpLeg), norm(lULDir))
    const lULLocal = quatMul(quatConj(hipsRot), quatMul(lULRot, hipsRot))
    vals.push(...quatToZXY(lULLocal))
    const lULWorld = quatMul(hipsRot, lULLocal)

    const lLDir    = sub(p.leftFoot, p.leftLeg)
    const lLRot    = quatFromTo(quatRotate(lULWorld, REST.leftLeg), norm(lLDir))
    const lLLocal  = quatMul(quatConj(lULWorld), quatMul(lLRot, lULWorld))
    vals.push(...quatToZXY(lLLocal))
    const lLWorld = quatMul(lULWorld, lLLocal)

    const lFDir    = sub(p.leftToeBase, p.leftFoot)
    const lFRot    = quatFromTo(quatRotate(lLWorld, REST.leftFoot), norm(lFDir))
    const lFLocal  = quatMul(quatConj(lLWorld), quatMul(lFRot, lLWorld))
    let lFEuler = quatToZXY(lFLocal)
    // Different limits, wider for turned-out feet, tighter for straight feet
    lFEuler[0]     = Math.max(-p.lFootRollLimit,  Math.min(p.lFootRollLimit,  lFEuler[0]))
    lFEuler[2]     = Math.max(-p.lFootTwistLimit, Math.min(p.lFootTwistLimit, lFEuler[2]))
    vals.push(...lFEuler)
    vals.push(0, 0, 0)

    // ── Right leg ─────────────────────────────────────────────────────────
    const rULDir   = sub(p.rightLeg, p.rightUpLeg) // Constrained knee
    const rULRot   = quatFromTo(quatRotate(hipsRot, REST.rightUpLeg), norm(rULDir))
    const rULLocal = quatMul(quatConj(hipsRot), quatMul(rULRot, hipsRot))
    vals.push(...quatToZXY(rULLocal))
    const rULWorld = quatMul(hipsRot, rULLocal)

    const rLDir    = sub(p.rightFoot, p.rightLeg)
    const rLRot    = quatFromTo(quatRotate(rULWorld, REST.rightLeg), norm(rLDir))
    const rLLocal  = quatMul(quatConj(rULWorld), quatMul(rLRot, rULWorld))
    vals.push(...quatToZXY(rLLocal))
    const rLWorld  = quatMul(rULWorld, rLLocal)
    
    const rFDir   = sub(p.rightToeBase, p.rightFoot)
    const rFRot   = quatFromTo(quatRotate(rLWorld, REST.rightFoot), norm(rFDir))
    const rFLocal = quatMul(quatConj(rLWorld), quatMul(rFRot, rLWorld))
    let rFEuler   = quatToZXY(rFLocal)
    // Different limits, wider for turned-out feet, tighter for straight feet
    rFEuler[0]    = Math.max(-p.rFootRollLimit,  Math.min(p.rFootRollLimit,  rFEuler[0]))
    rFEuler[2]    = Math.max(-p.rFootTwistLimit, Math.min(p.rFootTwistLimit, rFEuler[2]))
    vals.push(...rFEuler)
    vals.push(0, 0, 0)

    lines.push(vals.map(v => f(v)).join(' '))
  }

  return lines.join('\n')
}

// ── Public ────────────────────────────────────────────────────────────────────
export function exportBVH(frames, captureFps) {
  if (!frames?.length) return
  const off = getRestOffsets()
  const smoothedFrames = smoothFrames(frames, 3)
  const bvh = buildHierarchy(off) + '\n' + buildMotion(smoothedFrames, 1 / captureFps, off)
  const blob = new Blob([bvh], { type: 'text/plain' })
  const url  = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'pose_refined.bvh'; a.click()
  URL.revokeObjectURL(url)
}