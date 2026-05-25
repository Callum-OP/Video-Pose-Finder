// ── FABRIK IK Solver ──────────────────────────────────────────────────────────
// Forward And Backward Reaching Inverse Kinematics
//
// Takes raw MediaPipe world landmark positions and returns anatomically valid
// joint positions suitable for BVH export.
//
// Pipeline per frame:
//   1. Extract joint world positions from landmarks
//   2. Fit to fixed bone lengths (prevent skeleton stretching)
//   3. FABRIK solve each limb chain with pole vector hint
//   4. Apply joint angle constraints (knees forward, elbows back)
//
// References:
//  Aristidou & Lasenby (2011) "FABRIK: A fast, iterative solver for the IK problem"

// ── Vector maths ──────────────────────────────────────────────────────────────
const v = {
  add:   (a, b)    => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  sub:   (a, b)    => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  scale: (a, s)    => [a[0]*s, a[1]*s, a[2]*s],
  dot:   (a, b)    => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  len:   (a)       => Math.sqrt(a[0]**2 + a[1]**2 + a[2]**2),
  norm:  (a)       => { const l = Math.sqrt(a[0]**2+a[1]**2+a[2]**2)||1e-8; return [a[0]/l,a[1]/l,a[2]/l] },
  lerp:  (a, b, t) => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t],
  cross: (a, b)    => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  dist:  (a, b)    => Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2),
  // Move point 'a' toward point 'b' so it is exactly 'len' away from 'b'
  toward:(a, b, length) => {
    const dir = v.norm(v.sub(a, b))
    return v.add(b, v.scale(dir, length))
  }
}

// ── Bone length measurement ───────────────────────────────────────────────────
// Measured from first frame, locked for all subsequent frames, to prevent the skeleton from stretching or compressing.
function measureBoneLengths(positions) {
  const d = (a, b) => v.dist(positions[a], positions[b])
  return {
    // Spine
    spineLen:    d('hips',    'spine2') / 3,
    neckLen:     d('spine2',  'neck'),
    headLen:     d('neck',    'head'),
    // Left arm
    lUpperArm:   d('lShoulder', 'lElbow'),
    lForeArm:    d('lElbow',    'lWrist'),
    // Right arm
    rUpperArm:   d('rShoulder', 'rElbow'),
    rForeArm:    d('rElbow',    'rWrist'),
    // Left leg
    lThigh:      d('lHip',    'lKnee'),
    lShin:       d('lKnee',   'lAnkle'),
    // Right leg
    rThigh:      d('rHip',    'rKnee'),
    rShin:       d('rKnee',   'rAnkle'),
    // Shoulder width and hip width
    shoulderW:   d('lShoulder', 'rShoulder'),
    hipW:        d('lHip',      'rHip'),
  }
}

// ── Extract positions from MediaPipe world landmarks ──────────────────────────
function extractPositions(lms) {
  const p = (idx) => {
    const l = lms[idx]
    if (!l) return [0, 0, 0]
    // World landmarks: Y down → flip Y. Keep Z (toward camera = forward)
    return [l.x, -l.y, l.z]
  }
  const lHip   = p(23), rHip  = p(24)
  const lSho   = p(11), rSho  = p(12)
  const hips   = v.scale(v.add(lHip, rHip), 0.5)
  const sho    = v.scale(v.add(lSho, rSho), 0.5)
  return {
    hips,
    spine:      v.lerp(hips, sho, 0.33),
    spine1:     v.lerp(hips, sho, 0.66),
    spine2:     sho,
    neck:       v.lerp(sho, p(0), 0.5),
    head:       p(0),
    lShoulder:  lSho,
    lElbow:     p(13),
    lWrist:     p(15),
    rShoulder:  rSho,
    rElbow:     p(14),
    rWrist:     p(16),
    lHip,
    lKnee:      p(25),
    lAnkle:     p(27),
    rHip,
    rKnee:      p(26),
    rAnkle:     p(28),
  }
}

// ── FABRIK solver ─────────────────────────────────────────────────────────────
// Solves a 3-joint chain (root, mid, end) to reach a target.
// uses the poleTarget and length of both bones to determine bend direction, then applies optional hinge constraints.
// Returns: { root, mid, end } — solved positions

function fabrik3(root, mid, end, target, poleTarget, boneLenA, boneLenB, maxIter = 10) {
  const tolerance   = 0.001
  const totalLen    = boneLenA + boneLenB
  const targetDist  = v.dist(root, target)

  // If target is out of reach, stretch toward it
  if (targetDist > totalLen) {
    const dir = v.norm(v.sub(target, root))
    return {
      root,
      mid: v.add(root, v.scale(dir, boneLenA)),
      end: v.add(root, v.scale(dir, totalLen)),
    }
  }

  let positions = [
    [...root],
    [...mid],
    [...end],
  ]

  for (let iter = 0; iter < maxIter; iter++) {
    // ── Forward pass (end → root) ──────────────────────────────────────
    positions[2] = [...target]
    positions[1] = v.toward(positions[1], positions[2], boneLenB)
    positions[0] = v.toward(positions[0], positions[1], boneLenA)

    // ── Backward pass (root → end) ─────────────────────────────────────
    positions[0] = [...root]  // root is fixed
    positions[1] = v.toward(positions[1], positions[0], boneLenA)
    positions[2] = v.toward(positions[2], positions[1], boneLenB)

    if (v.dist(positions[2], target) < tolerance) break
  }

  // ── Pole vector constraint ─────────────────────────────────────────────
  // After FABRIK converges, rotate the mid joint around the root→end axis so it points toward the pole target.
  if (poleTarget) {
    const rootToEnd = v.norm(v.sub(positions[2], positions[0]))
    const midCurrent = positions[1]

    // Project mid joint onto the root→end axis
    const toMid = v.sub(midCurrent, positions[0])
    const projLen = v.dot(toMid, rootToEnd)
    const projPt = v.add(positions[0], v.scale(rootToEnd, projLen))

    // Project pole target onto the same axis
    const toPole = v.sub(poleTarget, positions[0])
    const poleProjL = v.dot(toPole, rootToEnd)
    const poleProjPt= v.add(positions[0], v.scale(rootToEnd, poleProjL))

    // Vectors from projection point to mid and to pole (both perp to bone axis)
    const midPerp = v.norm(v.sub(midCurrent, projPt))
    const polePerp = v.norm(v.sub(poleTarget, poleProjPt))

    if (v.len(v.sub(poleTarget, poleProjPt)) > 0.001) {
      // Angle between them
      const cosA = Math.max(-1, Math.min(1, v.dot(midPerp, polePerp)))
      const angle = Math.acos(cosA)

      if (Math.abs(angle) > 0.01) {
        // Rotation axis = bone axis (or flip if cross product says so)
        const cross = v.cross(midPerp, polePerp)
        const sign = v.dot(cross, rootToEnd) >= 0 ? 1 : -1
        const sinA = Math.sin(sign * angle)
        const cosAv = Math.cos(sign * angle)

        // Rodrigues rotation formula for rotating the mid point around the bone axis
        const ax = rootToEnd
        const toMidV = v.sub(midCurrent, projPt)
        const rotated = v.add(
          v.add(
            v.scale(toMidV, cosAv),
            v.scale(v.cross(ax, toMidV), sinA)
          ),
          v.scale(ax, v.dot(ax, toMidV) * (1 - cosAv))
        )
        positions[1] = v.add(projPt, rotated)

        // Re-enforce bone lengths after rotation
        positions[1] = v.toward(positions[1], positions[0], boneLenA)
        positions[2] = v.toward(positions[2], positions[1], boneLenB)
        // Snap end back to target
        positions[2] = v.toward(target, positions[1], boneLenB)
      }
    }
  }

  return { root: positions[0], mid: positions[1], end: positions[2] }
}

// ── Joint angle constraint ────────────────────────────────────────────────────
// Clamps the bend angle of a hinge joint (knee/elbow).
// minDeg: minimum bend (0 = fully straight)
// maxDeg: maximum bend (180 = fully folded back on itself)
// Returns corrected mid position.

function clampHinge(root, mid, end, minDeg, maxDeg, boneLenA, boneLenB) {
  const thigh  = v.norm(v.sub(mid, root))
  const shin   = v.norm(v.sub(end, mid))
  const cosA   = Math.max(-1, Math.min(1, v.dot(thigh, shin)))
  const angleDeg = (Math.acos(cosA) * 180 / Math.PI)
  // Bend angle = 180 - angle between segments (0 = straight, 180 = fully bent)
  const bendDeg  = 180 - angleDeg

  if (bendDeg >= minDeg && bendDeg <= maxDeg) return mid

  const targetBend = Math.max(minDeg, Math.min(maxDeg, bendDeg))
  const targetAngle = (180 - targetBend) * Math.PI / 180

  // Rotate shin around the axis perpendicular to the plane of the chain
  const planeNorm = v.norm(v.cross(thigh, shin))
  if (v.len(planeNorm) < 0.001) return mid  // degenerate case

  // Place mid at correct angle from root
  // Use Rodrigues to rotate thigh by targetAngle around planeNorm
  const rotShin = v.add(
    v.add(
      v.scale(thigh, Math.cos(targetAngle)),
      v.scale(v.cross(planeNorm, thigh), Math.sin(targetAngle))
    ),
    v.scale(planeNorm, v.dot(planeNorm, thigh) * (1 - Math.cos(targetAngle)))
  )

  const newMid = v.add(root, v.scale(rotShin, boneLenA))
  return newMid
}

// ── Main solver ───────────────────────────────────────────────────────────────
// Takes a frame's worldLandmarks, returns IK-solved positions in the same coordinate space, ready to be used/exported.

let cachedBoneLengths = null // Reused for all subsequent frames to prevent stretching

export function solveIK(worldLandmarks, isFirstFrame = false) {
  if (!worldLandmarks) return null

  const pos = extractPositions(worldLandmarks)

  // Measure bone lengths from first frame and lock them
  if (isFirstFrame || !cachedBoneLengths) {
    cachedBoneLengths = measureBoneLengths(pos)
  }
  const bl = cachedBoneLengths

  // ── Pole vectors ────────────────────────────────────────────────────────
  // Knees: pole points forward (in front of the body) — knees bend forward
  const kneePoleOffset  = [0, 0, 0.3]  // forward in world space
  const lKneePole = v.add(pos.lKnee,   kneePoleOffset)
  const rKneePole = v.add(pos.rKnee,   kneePoleOffset)

  // Elbows: pole points backward and outward — elbows bend behind
  const lElbowPole = v.add(pos.lElbow, [-0.2, 0, -0.3])
  const rElbowPole = v.add(pos.rElbow, [ 0.2, 0, -0.3])

  // ── Solve limbs ──────────────────────────────────────────────────────────

  // Left arm: includes shoulder, elbow, wrist
  const lArm = fabrik3(
    pos.lShoulder, pos.lElbow, pos.lWrist,
    pos.lWrist, // Target is the actual wrist landmark
    lElbowPole,
    bl.lUpperArm, bl.lForeArm
  )

  // Right arm
  const rArm = fabrik3(
    pos.rShoulder, pos.rElbow, pos.rWrist,
    pos.rWrist,
    rElbowPole,
    bl.rUpperArm, bl.rForeArm
  )

  // Left leg: includes hip, knee, ankle
  const lLeg = fabrik3(
    pos.lHip, pos.lKnee, pos.lAnkle,
    pos.lAnkle, // Target is the actual ankle landmark
    lKneePole,
    bl.lThigh, bl.lShin
  )

  // Right leg
  const rLeg = fabrik3(
    pos.rHip, pos.rKnee, pos.rAnkle,
    pos.rAnkle,
    rKneePole,
    bl.rThigh, bl.rShin
  )

  // ── Apply joint constraints ───────────────────────────────────────────────
  // Knees: can only bend forward (0°–150°), no hyperextension
  const lKneeClamped = clampHinge(lLeg.root, lLeg.mid, lLeg.end, 0, 150, bl.lThigh, bl.lShin)
  const rKneeClamped = clampHinge(rLeg.root, rLeg.mid, rLeg.end, 0, 150, bl.rThigh, bl.rShin)

  // Elbows: can bend 0°–160°
  const lElbowClamped = clampHinge(lArm.root, lArm.mid, lArm.end, 0, 160, bl.lUpperArm, bl.lForeArm)
  const rElbowClamped = clampHinge(rArm.root, rArm.mid, rArm.end, 0, 160, bl.rUpperArm, bl.rForeArm)

  // ── Return solved positions as pseudo-landmarks ───────────────────────────
  // We return an array in the same format as worldLandmarks so the BVH
  // exporter can use it without changes — just replace the relevant indices.
  const solved = worldLandmarks.map(lm => ({ ...lm }))  // shallow copy

  // Helper to write back — world space (un-flip Y since exporter flips it)
  const write = (idx, [x, y, z]) => {
    solved[idx] = { ...solved[idx], x, y: -y, z }
  }

  // Arms
  write(11, lArm.root)
  write(13, lElbowClamped)
  write(15, lArm.end)
  write(12, rArm.root)
  write(14, rElbowClamped)
  write(16, rArm.end)

  // Legs
  write(23, lLeg.root)
  write(25, lKneeClamped)
  write(27, lLeg.end)
  write(24, rLeg.root)
  write(26, rKneeClamped)
  write(28, rLeg.end)

  return solved
}

// Reset bone length cache, to be called when a new video is processed
export function resetIKCache() {
  cachedBoneLengths = null
}