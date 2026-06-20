// ── Pose cleanup ──────────────────────────────────────────────────────────────
// Post-processing that runs over the full captured sequence (after MediaPipe
// detection, before the One Euro filter).

// ── Temporal gap-fill for occluded joints ─────────────────────────────────────
// MediaPipe drops a joint's visibility when it goes behind the body or off-frame.
// Rather than leave those frames with stale/garbage positions, we interpolate each
// low-visibility joint from the
// nearest confident frames on either side. This also hands the One Euro filter a
// uniform, continuous series for the first time (previously low-confidence joints
// produced jumps the filter smeared across neighbours).
//
// Decision signal is the SCREEN landmark visibility (`landmarks[i].v`) — world
// landmarks carry no reliable per-joint visibility. When a joint is filled we
// interpolate BOTH the screen and world landmark positions (the BVH exporter
// reads world landmarks), but deliberately leave `v` untouched (low) so the
// median-bone-length pass keeps ignoring interpolated joints.
//
// Boundary behaviour:
//   - gap with a confident frame on both sides → linear interpolation
//   - gap at the very start/end (only one side) → hold the nearest value
//   - joint never confident in the whole clip → left as-is (exportBVH's
//     bone-length synthesis handles the fully-missing case)
// All interpolated values are guarded against NaN/Infinity, which the IIR One
// Euro filter would otherwise smear across every subsequent frame.
export function temporalGapFill(frames, confidenceThreshold, numLandmarks = 33) {
  if (!frames?.length) return frames
  const N = frames.length

  // Shallow-clone the per-frame landmark arrays so fills don't mutate originals.
  const outLms   = frames.map((f) => (f.landmarks ? [...f.landmarks] : f.landmarks))
  const outWorld = frames.map((f) => (f.worldLandmarks ? [...f.worldLandmarks] : f.worldLandmarks))

  const isConfident = (j, i) => {
    const lm = frames[j].landmarks?.[i]
    return lm && (lm.v ?? 1) >= confidenceThreshold
  }

  const lerp = (a, b, t) => a + (b - a) * t

  // Interpolate one landmark of one frame between two source frames, writing into
  // the cloned output array if all three components are finite.
  const fillPoint = (outArr, srcArrName, j, i, srcA, srcB, frac) => {
    if (!outArr[j]) return
    const a = frames[srcA][srcArrName]?.[i]
    const b = frames[srcB][srcArrName]?.[i]
    if (!a || !b) return
    const x = lerp(a.x, b.x, frac)
    const y = lerp(a.y, b.y, frac)
    const z = lerp(a.z, b.z, frac)
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return
    outArr[j][i] = { ...(outArr[j][i] || {}), x, y, z }
  }

  for (let i = 0; i < numLandmarks; i++) {
    // Nearest confident frame index at or before / at or after each position.
    const prev = new Array(N).fill(-1)
    const next = new Array(N).fill(-1)
    let last = -1
    for (let j = 0; j < N; j++) { if (isConfident(j, i)) last = j; prev[j] = last }
    let nxt = -1
    for (let j = N - 1; j >= 0; j--) { if (isConfident(j, i)) nxt = j; next[j] = nxt }

    for (let j = 0; j < N; j++) {
      if (isConfident(j, i)) continue
      const p = prev[j]; const n = next[j]

      let srcA; let srcB; let frac
      if (p >= 0 && n >= 0 && p !== n) { srcA = p; srcB = n; frac = (j - p) / (n - p) }
      else if (p >= 0) { srcA = p; srcB = p; frac = 0 }   // trailing gap → hold last
      else if (n >= 0) { srcA = n; srcB = n; frac = 0 }   // leading gap → hold first
      else continue                                       // never confident → leave for exporter

      fillPoint(outLms,   'landmarks',      j, i, srcA, srcB, frac)
      fillPoint(outWorld, 'worldLandmarks', j, i, srcA, srcB, frac)
    }
  }

  return frames.map((f, j) => ({ ...f, landmarks: outLms[j], worldLandmarks: outWorld[j] }))
}
