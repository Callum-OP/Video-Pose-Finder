// ── One Euro Filter ────────────────────────────────────────────────────
// Implements the 1€ Filter (Casiez et al. 2012) for 3D landmark streams.

class LowPassFilter {
  constructor(initValue = 0) {
    this.y = initValue
    this.s = initValue
    this._initialized = false
  }

  filter(x, alpha) {
    if (!this._initialized) {
      this.y = x
      this.s = x
      this._initialized = true
      return x
    }
    this.y = alpha * x + (1 - alpha) * this.s
    this.s = this.y
    return this.y
  }

  get lastValue() { return this.s }
}

// One Euro Filter for a single scalar channel
class OneEuroScalar {
  // beta is the speed coefficient: higher = less smoothing during fast motion, so
  // quick action (flips, falls) tracks instead of lagging/rounding off. 0.02 is a
  // moderate bump from the old 0.007 — raise further if fast motion still lags,
  // lower toward 0.007 if slow footage looks jittery.
  constructor(freq, fmin = 1.0, beta = 0.02, dcutoff = 1.0) {
    this.freq    = freq
    this.fmin    = fmin
    this.beta    = beta
    this.dcutoff = dcutoff
    this.xFilter = new LowPassFilter()
    this.dxFilter = new LowPassFilter()
    this._initialized = false
    this._lastValue = 0
  }

  _alpha(cutoff) {
    const te = 1 / this.freq
    const tau = 1 / (2 * Math.PI * cutoff)
    return 1 / (1 + tau / te)
  }

  filter(x) {
    // Estimate derivative
    const dx = this._initialized ? (x - this._lastValue) * this.freq : 0
    this._initialized = true
    this._lastValue = x

    const edx = this.dxFilter.filter(dx, this._alpha(this.dcutoff))
    const cutoff = this.fmin + this.beta * Math.abs(edx)
    return this.xFilter.filter(x, this._alpha(cutoff))
  }

  reset() {
    this.xFilter = new LowPassFilter()
    this.dxFilter = new LowPassFilter()
    this._initialized = false
  }
}

// Filter for a full 3D point {x, y, z}
class OneEuroPoint3D {
  constructor(freq, fmin = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.fx = new OneEuroScalar(freq, fmin, beta, dcutoff)
    this.fy = new OneEuroScalar(freq, fmin, beta, dcutoff)
    this.fz = new OneEuroScalar(freq, fmin, beta, dcutoff)
  }

  filter({ x, y, z }) {
    return {
      x: this.fx.filter(x),
      y: this.fy.filter(y),
      z: this.fz.filter(z),
    }
  }

  reset() { this.fx.reset(); this.fy.reset(); this.fz.reset() }
}

// ── Filter bank for all 33 MediaPipe landmarks ───────────────────────────────
// One bank instance per person track. Call .filter(frame) on each captured frame, in chronological order. 
export class LandmarkFilterBank {
  constructor({
    freq     = 30, // This is replaced later in usePoseExtractor with the actual capture FPS
    fmin     = 1.0,
    beta     = 0.02, // speed coefficient — higher tracks fast motion better (was 0.007)
    dcutoff  = 1.0,
    numLandmarks = 33
  } = {}) {
    this.filters = Array.from(
      { length: numLandmarks },
      () => new OneEuroPoint3D(freq, fmin, beta, dcutoff)
    )
    this.worldFilters = Array.from(
      { length: numLandmarks },
      () => new OneEuroPoint3D(freq, fmin, beta, dcutoff)
    )
  }

  // Returns a new frame with filtered landmark arrays
  filter(frame) {
    const filteredLandmarks = frame.landmarks.map((lm, i) =>
      lm ? { ...lm, ...this.filters[i].filter(lm) } : lm
    )

    const filteredWorldLandmarks = frame.worldLandmarks
      ? frame.worldLandmarks.map((lm, i) =>
          lm ? { ...lm, ...this.worldFilters[i].filter(lm) } : lm
        )
      : null

    return { ...frame, landmarks: filteredLandmarks, worldLandmarks: filteredWorldLandmarks }
  }

  // Can be called to be reset if camera cut or person track change happens
  reset() {
    this.filters.forEach(f => f.reset())
    this.worldFilters.forEach(f => f.reset())
  }
}