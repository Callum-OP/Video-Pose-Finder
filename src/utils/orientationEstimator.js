// ── Orientation Estimator ─────────────────────────────────────────────
// Find out the orientation of a pose and if the camera angle has changed suddenly

const DEG = 180 / Math.PI
const IDX = { lSho: 11, rSho: 12, lHip: 23, rHip: 24, lElbow: 13, rElbow: 14, lKnee: 25, rKnee: 26 }

function sub(a, b)  { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
function len(v)     { return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) }
function norm(v)    { const l = len(v)||1e-8; return [v[0]/l, v[1]/l, v[2]/l] }
function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] }
function wlm(lm)    { return lm ? [lm.x, lm.y, lm.z] : null }

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

function visibilityAsymmetryYaw(lms) {
  const pairs = [[IDX.lElbow, IDX.rElbow], [IDX.lKnee, IDX.rKnee]]
  let lv = 0, rv = 0
  for (const [li, ri] of pairs) { lv += lms[li]?.v ?? 0; rv += lms[ri]?.v ?? 0 }
  const asym = (lv - rv) / (lv + rv + 1e-6)
  return asym * 90
}

export class OrientationEstimator {
  constructor({ captureFps = 30 } = {}) {
    this._freq     = captureFps
    this._prevYaw  = null
    this._yawSmooth = 0
    // Simple low-pass for yaw
    this._alpha = 0.3
  }

  process(frame) {
    const { landmarks: lms, worldLandmarks: worldLms } = frame

    const s2 = hipAzimuthYaw(worldLms)
    const s3 = visibilityAsymmetryYaw(lms)

    const rawYaw = s2 !== null
      ? (s3 !== null ? s2 * 0.75 + s3 * 0.25 : s2)
      : (s3 ?? 0)

    // If yaw jump > 90° in one frame then it's probably a shot cut
    let shotCut = false
    if (this._prevYaw !== null) {
      let delta = rawYaw - this._prevYaw
      if (delta >  180) delta -= 360
      if (delta < -180) delta += 360
      if (Math.abs(delta) > 90) {
        shotCut = true
        this._yawSmooth = rawYaw  // Snap instead of blending
      }
    } else {
      this._yawSmooth = rawYaw
    }

    this._yawSmooth = this._alpha * rawYaw + (1 - this._alpha) * this._yawSmooth
    this._prevYaw   = rawYaw

    const absYaw = Math.abs(this._yawSmooth)
    const view =
      absYaw < 45  ? 'front' :
      absYaw < 135 ? 'side'  :
      absYaw < 160 ? 'rear_3q' : 'rear'

    return {
      ...frame,
      orientation: { yaw: this._yawSmooth, rawYaw, view, shotCut }
    }
  }

  reset() { this._prevYaw = null; this._yawSmooth = 0 }
}