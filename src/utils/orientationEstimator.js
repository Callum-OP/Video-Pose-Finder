// ── Orientation Estimator ─────────────────────────────────────────────────────
// Estimates body yaw (rotation around Y axis) from MediaPipe landmarks.
// Fuses multiple geometric signals into a single smoothed yaw, and detects shot cuts.
//
// Yaw convention: 0° = facing camera, 90° = person's left side facing camera, -90° = person's right side, 180°/-180° = facing away.

const DEG = 180 / Math.PI
const IDX = {
  nose:   0,
  lEar:   7,  rEar:   8,
  lSho:  11,  rSho:  12,
  lElbow:13,  rElbow:14,
  lHip:  23,  rHip:  24,
  lKnee: 25,  rKnee: 26,
}

// ── Math helpers (plain arrays [x,y,z]) ─────────────────────────────
function sub(a, b)  { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
function dot(a, b)  { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
function len(v)     { return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) }
function norm(v)    { const l = len(v)||1e-8; return [v[0]/l, v[1]/l, v[2]/l] }
function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] }
function wlm(lm)    { return lm ? [lm.x, lm.y, lm.z] : null }

// ── Hip azimuth from world landmarks ────────────────────────────────
// Cross product of hip-right and spine-up vectors gives the body's forward direction.
// Most reliable signal at large rotation angles.
function hipAzimuthYaw(worldLms) {
  const lh = wlm(worldLms?.[IDX.lHip]),  rh = wlm(worldLms?.[IDX.rHip])
  const ls = wlm(worldLms?.[IDX.lSho]),  rs = wlm(worldLms?.[IDX.rSho])
  if (!lh || !rh || !ls || !rs) return null
  const hipMid = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
  const shoMid = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
  const up       = norm(sub(shoMid, hipMid))
  const hipRight = norm(sub(rh, lh))
  const fwd      = norm(cross(hipRight, up))
  return Math.atan2(fwd[0], fwd[2]) * DEG
}

// ── Visibility asymmetry ───────────────────────────────────────────
// When the body turns, far-side joints (elbow, knee) lose visibility.
// The asymmetry tells us which direction the turn is.
function visibilityAsymmetryYaw(lms) {
  const pairs = [[IDX.lElbow, IDX.rElbow], [IDX.lKnee, IDX.rKnee]]
  let lv = 0, rv = 0
  for (const [li, ri] of pairs) { lv += lms[li]?.v ?? 0; rv += lms[ri]?.v ?? 0 }
  const asym = (lv - rv) / (lv + rv + 1e-6)
  return asym * 90
}

// ── Nose visibility (frontal confidence) ────────────────────────────
// Nose visibility drops as the person turns away from camera.
// Used as a bias toward 0° when the person is clearly front-facing.
function noseFrontalConfidence(lms) {
  return lms[IDX.nose]?.v ?? 0
}

// ── Ear separation + asymmetry ─────────────────────────────────────
// Ear pixel separation narrows when side-on (ears align front-to-back).
// Ear visibility asymmetry tells which direction they turned:
//   Left ear more visible  → person turned to right (positive yaw)
//   Right ear more visible → person turned to left (negative yaw)
function earSignals(lms) {
  const lEar = lms[IDX.lEar], rEar = lms[IDX.rEar]
  if (!lEar || !rEar) return { yaw: null, confidence: 0 }
  const asym = (lEar.v - rEar.v) / (lEar.v + rEar.v + 1e-6)
  const sep = Math.abs(rEar.x - lEar.x)
  // Low separation = side-on = high yaw magnitude
  const absYawEst = Math.max(0, Math.min(90, (0.12 - sep) / 0.12 * 90))
  const signedYaw = absYawEst * Math.sign(asym)
  const confidence = Math.min(1, Math.abs(asym) * 2)
  return { yaw: signedYaw, confidence }
}

// ── Shoulder depth asymmetry ───────────────────────────────────────
// In world landmarks, when turning, the near shoulder has a larger Z value than the far shoulder, Z difference is proportional to sin(yaw).
function shoulderDepthYaw(worldLms) {
  const ls = worldLms?.[IDX.lSho], rs = worldLms?.[IDX.rSho]
  if (!ls || !rs) return null
  const zDiff = ls.z - rs.z
  const shoulderWidth = Math.sqrt((rs.x-ls.x)**2 + (rs.y-ls.y)**2 + (rs.z-ls.z)**2)
  if (shoulderWidth < 0.05) return null
  const sinYaw = Math.max(-1, Math.min(1, zDiff / shoulderWidth))
  return -Math.asin(sinYaw) * DEG
}

// ── Weighted fusion of all geometric signal─────────────────────────
function fuseYawSignals(lms, worldLms) {
  const s1   = hipAzimuthYaw(worldLms)
  const s2   = visibilityAsymmetryYaw(lms)
  const s4   = earSignals(lms)
  const s5   = shoulderDepthYaw(worldLms)
  const nose = noseFrontalConfidence(lms)

  // When nose is clearly visible, bias the estimate toward front-facing.
  const frontBias = nose > 0.8 ? (1 - nose) * 30 : 0

  let weightedSum = 0, totalWeight = 0
  const add = (signal, weight) => {
    if (signal === null || signal === undefined) return
    weightedSum  += signal * weight
    totalWeight  += weight
  }

  add(s1,       0.35)                      
  add(s2,       0.10)                       
  add(s4.yaw,   s4.confidence * 0.30)       
  add(s5,       0.25)                       

  if (!totalWeight) return 0

  const raw = weightedSum / totalWeight
  return raw * (1 - frontBias / 90)
}

// ── Main Estimator Class ────────────────────────────────────────────
export class OrientationEstimator {
  constructor({ captureFps = 30 } = {}) {
    this._freq      = captureFps
    this._prevYaw   = null
    this._yawSmooth = 0
    this._alpha = 0.3
  }

  // ── Process one frame ─────────────────────────────────────────────────────
  // Returns the frame enriched with: { yaw, rawYaw, view, shotCut, source }.
  process(frame) {
    const { landmarks: lms, worldLandmarks: worldLms } = frame

    // ── Geometric yaw estimate from the landmarks ─────────────────────────
    const rawYaw = fuseYawSignals(lms, worldLms)

    // ── Shot cut detection ────────────────────────────────────────────────
    // A yaw jump > 90° in one frame is likely a camera cut or tracking failure and not a real rotation.
    // Reset the smoother to the new value so it doesn't try to blend between the two frames.
    let shotCut = false
    if (this._prevYaw !== null) {
      let delta = rawYaw - this._prevYaw
      if (delta >  180) delta -= 360
      if (delta < -180) delta += 360
      if (Math.abs(delta) > 90) {
        shotCut = true
        this._yawSmooth = rawYaw
      }
    } else {
      this._yawSmooth = rawYaw
    }

    this._yawSmooth = this._alpha * rawYaw + (1 - this._alpha) * this._yawSmooth
    this._prevYaw   = rawYaw

    const absYaw = Math.abs(this._yawSmooth)
    const view =
      absYaw < 45  ? 'front'   :
      absYaw < 135 ? 'side'    :
      absYaw < 160 ? 'rear_3q' : 'rear'

    return {
      ...frame,
      orientation: {
        yaw:     this._yawSmooth,
        rawYaw,
        view,
        shotCut,
        source:  'heuristic',
      }
    }
  }

  reset() {
    this._prevYaw   = null
    this._yawSmooth = 0
  }
}