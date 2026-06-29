import { useRef, useState, useCallback, useEffect } from 'react'
import { PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { LandmarkFilterBank } from '../utils/oneEuroFilter'
import { temporalGapFill } from '../utils/poseCleanup'

// ── Snapshot a video frame to an offscreen canvas ─────────────────────────────
// The key fix for the video-vs-image accuracy gap. Instead of passing the <video>
// element directly to MediaPipe (which reads whatever frame the decoder currently
// holds), we draw the current frame to a canvas first. The canvas is a stable
// snapshot that won't change mid-detection, matching what the image path does.
function snapshotVideoFrame(videoEl) {
  const canvas = document.createElement('canvas')
  canvas.width  = videoEl.videoWidth
  canvas.height = videoEl.videoHeight
  canvas.getContext('2d').drawImage(videoEl, 0, 0)
  return canvas
}

// MediaPipe landmark indices for the connections that'll be drawn.
export const POSE_CONNECTIONS = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
  // Face outline (simplified)
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
]

export const NAMED_JOINTS = {
  0:  'nose',
  11: 'l_shoulder', 12: 'r_shoulder',
  13: 'l_elbow',    14: 'r_elbow',
  15: 'l_wrist',    16: 'r_wrist',
  23: 'l_hip',      24: 'r_hip',
  25: 'l_knee',     26: 'r_knee',
  27: 'l_ankle',    28: 'r_ankle',
}

// These are the default settings the user will be able to adjust them in the UI later.
export const DEFAULT_SETTINGS = {
  captureFps:          30,    // Samples per second taken from the video.
  confidenceThreshold: 0.5,   // Min accepted amount of visibility to keep a frame/pose.
  keyframeThreshold:   0.04,  // Min accepted amount of joint movement (0–1 normalised) to keep a frame/pose.
  maxFrames:           200,   // The number of frames/poses to keep.
  modelQuality:        'full', // Pose model: 'lite' | 'full' | 'heavy' — higher = more accurate, slower.
  trackHands:          true,   // Run the hand/finger pipeline. Off = faster (skips a per-frame model pass).
  // ── Export-time options (read at BVH export, applied without reprocessing) ──
  keepFeetPlanted:     true,   // Ground the feet to a stable floor. Off = feet follow raw motion (better for aerial/action).
  strictAnatomy:       false,  // Tighten joint limits (foot roll/twist, wrist flex) so limbs stay more rigid/neutral.
  preserveFacing:      true,   // Root carries the body's turning/facing. Off = facing stabilised (root yaw locked).
}

export const PERSON_COLORS = ['#7c6cff', '#39e8a0', '#f5a623', '#ff4d6d']

// Pose landmarker model variants. Heavy is the most accurate, lite the fastest.
export const MODEL_URLS = {
  lite:  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  full:  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}
const poseModelUrl = (quality) => MODEL_URLS[quality] ?? MODEL_URLS.full

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

const KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

// ── MediaPipe hand landmark indices ──────────────────────────────────────────
// 0: wrist
// 1-4:   thumb  (CMC, MCP, IP, tip)
// 5-8:   index  (MCP, PIP, DIP, tip)
// 9-12:  middle (MCP, PIP, DIP, tip)
// 13-16: ring   (MCP, PIP, DIP, tip)
// 17-20: pinky  (MCP, PIP, DIP, tip)
export const HAND_FINGER_INDICES = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  pinky:  [17, 18, 19, 20],
}

function avgConfidence(landmarks) {
  const sum = KEY_JOINT_INDICES.reduce((acc, i) => acc + (landmarks[i]?.v ?? 0), 0)
  return sum / KEY_JOINT_INDICES.length
}

function poseDiff(landmarksA, landmarksB) {
  let total = 0
  for (const i of KEY_JOINT_INDICES) {
    const a = landmarksA[i]; const b = landmarksB[i]
    if (!a || !b) continue
    total += Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
  }
  return total / KEY_JOINT_INDICES.length
}

function subsampleFrames(frames, maxFrames) {
  if (frames.length <= maxFrames) return frames
  const step = (frames.length - 1) / (maxFrames - 1)
  return Array.from({ length: maxFrames }, (_, i) => frames[Math.round(i * step)])
}

// ── Robust per-bone lengths from the whole sequence ───────────────────────────
// MediaPipe landmark index pairs for each constrained bone, matching the keys
// getBoneLengths() returns in exportBVH.js.
const BONE_PAIRS = {
  lThigh: [23, 25], rThigh: [24, 26],
  lShin:  [25, 27], rShin:  [26, 28],
  lUpper: [11, 13], rUpper: [12, 14],
  lFore:  [13, 15], rFore:  [14, 16],
}
// Must match SCALE in exportBVH.js — bone length is rotation/sign invariant, so
// the only space difference between raw world landmarks and exportBVH's internal
// coordinates is this uniform ×100 scale.
const BONE_SCALE = 100

function median(values) {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

// Compute the median length of each bone over frames where both endpoints are
// confident, in exportBVH's scaled space. Replaces exportBVH's fragile
// "cache from the first frame" behaviour (the first frame is often the mangled
// one, which previously poisoned every limb length for the whole clip).
export function computeMedianBoneLengths(frames, confidenceThreshold) {
  const samples = {}
  for (const key of Object.keys(BONE_PAIRS)) samples[key] = []

  for (const frame of frames) {
    const w = frame.worldLandmarks
    if (!w) continue
    const lms = frame.landmarks
    for (const [key, [a, b]] of Object.entries(BONE_PAIRS)) {
      const wa = w[a]; const wb = w[b]
      if (!wa || !wb) continue
      const va = lms?.[a]?.v ?? 1; const vb = lms?.[b]?.v ?? 1
      if (va < confidenceThreshold || vb < confidenceThreshold) continue
      const dx = wa.x - wb.x; const dy = wa.y - wb.y; const dz = wa.z - wb.z
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) * BONE_SCALE
      if (len > 0 && Number.isFinite(len)) samples[key].push(len)
    }
  }

  const out = {}
  let any = false
  for (const key of Object.keys(BONE_PAIRS)) {
    const m = median(samples[key])
    if (m) { out[key] = m; any = true }
  }
  return any ? out : null
}

// Hip midpoint, used as the seed position for tracking.
export function hipCenter(landmarks) {
  const l = landmarks[23]; const r = landmarks[24]
  if (!l || !r) return null
  return { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 }
}

function dist2D(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) }

function normaliseLandmarks(rawLms) {
  return rawLms.map((lm) => ({
    x: parseFloat(lm.x.toFixed(4)),
    y: parseFloat(lm.y.toFixed(4)),
    z: parseFloat(lm.z.toFixed(4)),
    v: parseFloat((lm.visibility ?? 1).toFixed(3)),
  }))
}

// Normalise hand landmarks (no visibility field, use 1.0)
function normaliseHandLandmarks(rawLms) {
  return rawLms.map((lm) => ({
    x: parseFloat(lm.x.toFixed(4)),
    y: parseFloat(lm.y.toFixed(4)),
    z: parseFloat(lm.z.toFixed(4)),
    v: 1.0,
  }))
}

// ── Compute finger bend angles from MediaPipe hand landmarks ──────────────────
// For each finger, compute bend at each joint using the angle between consecutive bone vectors.
// Returns a per-finger structure the BVH exporter consumes for finger rotations.
function computeFingerAnglesFromLandmarks(handLms) {
  const sub  = (a, b) => [a.x - b.x, a.y - b.y, a.z - b.z]
  const dot  = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
  const len  = (v) => Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
  const norm = (v) => { const l = len(v) || 1e-8; return [v[0]/l, v[1]/l, v[2]/l] }

  function bendAngle(a, b, c) {
    const ba = norm(sub(a, b))
    const bc = norm(sub(c, b))
    const cosA = Math.max(-1, Math.min(1, dot(ba, bc)))
    // Interior angle at joint b; 180° = straight, 90° = bent 90°
    return 180 - Math.acos(cosA) * (180 / Math.PI)
  }

  const lm = handLms  // shorthand

  return {
    thumb: {
      mcp: bendAngle(lm[1], lm[2], lm[3]),
      ip:  bendAngle(lm[2], lm[3], lm[4]),
    },
    index: {
      mcp: bendAngle(lm[0], lm[5], lm[6]),
      pip: bendAngle(lm[5], lm[6], lm[7]),
      dip: bendAngle(lm[6], lm[7], lm[8]),
    },
    middle: {
      mcp: bendAngle(lm[0], lm[9],  lm[10]),
      pip: bendAngle(lm[9], lm[10], lm[11]),
      dip: bendAngle(lm[10],lm[11], lm[12]),
    },
    ring: {
      mcp: bendAngle(lm[0], lm[13], lm[14]),
      pip: bendAngle(lm[13],lm[14], lm[15]),
      dip: bendAngle(lm[14],lm[15], lm[16]),
    },
    pinky: {
      mcp: bendAngle(lm[0], lm[17], lm[18]),
      pip: bendAngle(lm[17],lm[18], lm[19]),
      dip: bendAngle(lm[18],lm[19], lm[20]),
    },
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

// Tri-state cache for whether requestVideoFrameCallback actually fires on our
// offscreen processing <video>: null = unprobed, true = works, false = times out.
// A detached (never-in-DOM) video often never presents frames, so rVFC never
// fires; without this probe we'd pay the full rVFC timeout on EVERY frame, which
// dominates processing time on long clips. Capability is stable per browser, so
// one probe per session is enough.
let _rvfcWorks = null

// ── Seek a video and wait until the target frame is actually decoded ──────────
// Seeking alone does not guarantee the decoder has *painted* the seeked frame, so
// MediaPipe can read a stale/garbage frame — this is the root cause of the
// "first frame appears mangled" bug and a general source of per-frame error.
// We:
//   1. attach the `seeked` listener BEFORE setting currentTime (the old
//      `video.onseeked = r` assignment could miss an event fired too early),
//   2. race it against a 1s timeout so a missed event can never hang the loop,
//   3. await one requestVideoFrameCallback the FIRST time only as a probe; if it
//      doesn't fire we stop using it for the rest of the run (the `seeked` event
//      plus the canvas snapshot already guarantee frame stability).
// snapshotVideoFrame remains the real frame-stability guarantee; rVFC just
// maximises the chance the snapshot holds the seeked frame.
async function seekToFrame(video, t) {
  // At t=0 the decoder often hasn't painted yet, and seeking to the current time
  // (0) may not fire `seeked` at all — nudge to a tiny epsilon so it does.
  const target = t <= 0 ? Math.min(0.001, video.duration || 0.001) : t

  const seeked = new Promise((resolve) => {
    video.addEventListener('seeked', resolve, { once: true })
  })
  video.currentTime = target
  await Promise.race([seeked, sleep(1000)])

  // Skip rVFC once we've learned it never fires on this detached video.
  if (_rvfcWorks === false || typeof video.requestVideoFrameCallback !== 'function') return

  // Probe (longer budget) once; afterwards trust the cached result with a short
  // budget so a stray slow frame can't stall the loop.
  const budget = _rvfcWorks === null ? 250 : 50
  const fired = await Promise.race([
    new Promise((resolve) => video.requestVideoFrameCallback(() => resolve(true))),
    sleep(budget).then(() => false),
  ])
  if (_rvfcWorks === null) _rvfcWorks = fired
}

// ── Persist the last processed result ─────────────────────────────────────────
// The Export buttons regenerate the BVH/JSON from the in-memory `frames`, so we
// only persist `frames` + `stats` to restore a previous session after a reload.
// The source file isn't kept (Files can't be serialised, and Rescan needs the
// original upload), but the processed result — and therefore export — survives.
const RESULT_STORAGE_KEY = 'posefinder_last_result'

function loadSavedResult() {
  try {
    const raw = localStorage.getItem(RESULT_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.frames?.length) return parsed
  } catch {
    // Corrupt/unreadable entry — ignore and start fresh.
  }
  return null
}

// Deep-clone just the editable pose data of a frame (landmarks + hand/finger data).
// Used to keep an immutable "originally-captured" copy for the 3D editor's Reset.
function cloneFrameData(frame) {
  return {
    landmarks:      frame.landmarks ? frame.landmarks.map((lm) => ({ ...lm })) : frame.landmarks,
    worldLandmarks: frame.worldLandmarks ? frame.worldLandmarks.map((lm) => ({ ...lm })) : null,
    handData:       frame.handData ? structuredClone(frame.handData) : null,
  }
}

// Main Public Hook
export function usePoseExtractor() {
  const landmarkerRef     = useRef(null)
  const handLandmarkerRef = useRef(null)   // MediaPipe HandLandmarker
  const visionRef         = useRef(null)
  const scrubVideoRef     = useRef(null)
  const fileRef           = useRef(null)
  const isCancelledRef    = useRef(false)
  const filterBankRef     = useRef(null)

  // ── Screen wake lock ───────────────────────────────────────────────────────
  // Long runs (especially the heavy model) can take many minutes, during which
  // the laptop may auto-sleep and wipe all in-memory progress. We hold a screen
  // wake lock while processing to keep the machine awake. wantWakeLockRef tracks
  // whether we *should* be holding it, so we can re-acquire after the OS releases
  // it on tab switches.
  const wakeLockRef     = useRef(null)
  const wantWakeLockRef = useRef(false)

  // ── Monotonic timestamp counter ───────────────────────────────────────────
  // This is the primary fix for why video is less accurate than images.
  // MediaPipe's VIDEO mode requires strictly increasing timestamps. When seeking
  // through a video frame-by-frame, the decoder's reported time can be
  // non-monotonic or stall at the same value, causing MediaPipe to return stale
  // results from the previous frame. We maintain our own counter that always
  // increases, decoupled from actual video time.
  const syntheticTimestampRef = useRef(0)

  // Restore the last processed result (if any) so it's available after a reload.
  // useRef has no lazy initialiser, so guard with `undefined` to read storage once.
  const restoredRef = useRef(undefined)
  if (restoredRef.current === undefined) restoredRef.current = loadSavedResult()

  // Immutable "as-captured" copy of each frame's pose, aligned with the `frames`
  // array, so the 3D editor can reset a single frame. Seeded from the restored
  // result (falling back to the frames themselves for older saves) and refreshed
  // whenever a fresh run produces new frames.
  const originalFramesRef = useRef(undefined)
  if (originalFramesRef.current === undefined) {
    const r = restoredRef.current
    const src = r?.originalFrames ?? r?.frames ?? []
    originalFramesRef.current = src.map(cloneFrameData)
  }

  const [status, setStatus]           = useState(() => restoredRef.current ? 'done' : 'idle')
  const [progress, setProgress]       = useState(0)
  const [frames, setFrames]           = useState(() => restoredRef.current?.frames ?? [])
  const [stats, setStats]             = useState(() => restoredRef.current?.stats ?? null)
  const [error, setError]             = useState(null)
  const [duration, setDuration]       = useState(0)
  // Detected persons at the current scrub position
  const [scrubPersons, setScrubPersons]   = useState([])
  // Storage array for the filmstrip/pre-scan view mode
  const [scanFrames, setScanFrames]       = useState([])

  const selectPersonRef  = useRef(null)
  const statsSummaryRef  = useRef(null)

  const scrollToRef = (targetRef) => {
    setTimeout(() => {
      if (targetRef.current) {
        targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 150)
  }

  const acquireWakeLock = useCallback(async () => {
    wantWakeLockRef.current = true
    try {
      if ('wakeLock' in navigator) {
        wakeLockRef.current = await navigator.wakeLock.request('screen')
      }
    } catch {
      // Wake lock can be rejected (e.g. low battery) — non-fatal, processing continues.
    }
  }, [])

  const releaseWakeLock = useCallback(() => {
    wantWakeLockRef.current = false
    try { wakeLockRef.current?.release?.() } catch { /* already released */ }
    wakeLockRef.current = null
  }, [])

  // The OS releases the wake lock when the tab is hidden; re-acquire it when the
  // tab becomes visible again if we still want it (i.e. still processing).
  useEffect(() => {
    const onVisible = () => {
      if (wantWakeLockRef.current && document.visibilityState === 'visible' && !wakeLockRef.current) {
        acquireWakeLock()
      }
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [acquireWakeLock])

  // Persist the result whenever a run completes, so it can be restored on reload.
  // Only writes on a finished run (status 'done'); cancels/in-progress runs never
  // overwrite the saved copy.
  useEffect(() => {
    if (status !== 'done' || frames.length === 0) return
    try {
      localStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify({
        frames, stats, originalFrames: originalFramesRef.current, savedAt: Date.now(),
      }))
    } catch (e) {
      // Quota exceeded or non-serialisable — non-fatal, just skip persisting.
      console.warn('[PoseFinder] Could not save last result to localStorage:', e)
    }
  }, [status, frames, stats])

  // Clear the displayed result and the saved copy (the "clear" action).
  const clearResults = useCallback(() => {
    try { localStorage.removeItem(RESULT_STORAGE_KEY) } catch { /* ignore */ }
    originalFramesRef.current = []
    setFrames([])
    setStats(null)
    setError(null)
    setProgress(0)
    setStatus('idle')
  }, [])

  // ── 3D pose editor hooks ────────────────────────────────────────────────────
  // Commit an edited pose for a single frame (immutable update). `patch` may carry
  // any editable frame fields (landmarks, worldLandmarks, handData). The persistence
  // effect re-saves automatically because `frames` changes while status is 'done'.
  const applyFrameEdit = useCallback((frameIndex, patch) => {
    setFrames((prev) => prev.map((f, i) => (
      i === frameIndex ? { ...f, ...patch } : f
    )))
  }, [])

  // Restore a single frame to its originally-captured pose.
  const resetFrame = useCallback((frameIndex) => {
    const orig = originalFramesRef.current?.[frameIndex]
    if (!orig) return
    setFrames((prev) => prev.map((f, i) => (
      i === frameIndex
        ? { ...f, ...cloneFrameData(orig) }
        : f
    )))
  }, [])

  const cancelProcessing = useCallback(() => {
    isCancelledRef.current = true
    releaseWakeLock()
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    setScanFrames([])
    fileRef.current = null
    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }
    setStatus('idle')
  }, [releaseWakeLock])

  async function createLandmarker(numPoses, modelQuality = 'full') {
    if (!visionRef.current) {
      setStatus('loading-model')
      visionRef.current = await FilesetResolver.forVisionTasks(WASM_URL)
    }
    if (landmarkerRef.current) {
      landmarkerRef.current.close()
      landmarkerRef.current = null
    }
    const landmarker = await PoseLandmarker.createFromOptions(visionRef.current, {
      baseOptions: { modelAssetPath: poseModelUrl(modelQuality), delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses,
      minPoseDetectionConfidence: 0.4,
      minPosePresenceConfidence:  0.4,
      minTrackingConfidence:      0.4,
    })
    landmarkerRef.current = landmarker
    return landmarker
  }

  // ── Create MediaPipe HandLandmarker ───────────────────────────────────────
  async function createHandLandmarker() {
    if (!visionRef.current) {
      visionRef.current = await FilesetResolver.forVisionTasks(WASM_URL)
    }
    if (handLandmarkerRef.current) {
      handLandmarkerRef.current.close()
      handLandmarkerRef.current = null
    }
    const handLandmarker = await HandLandmarker.createFromOptions(visionRef.current, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands:                    2,
      minHandDetectionConfidence:  0.5,
      minHandPresenceConfidence:   0.5,
      minTrackingConfidence:       0.5,
    })
    handLandmarkerRef.current = handLandmarker
    return handLandmarker
  }

  // ── Load video and landmarker, enter scrub mode ────────────────────────────
  const loadVideo = useCallback(async (file) => {
    isCancelledRef.current = false
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    setScanFrames([])
    fileRef.current = file

    // Clean up any previous scrub video
    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    let landmarker
    try {
      landmarker = await createLandmarker(4)
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    const video = document.createElement('video')
    video.src = URL.createObjectURL(file)
    video.muted = true; video.playsInline = true
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })

    scrubVideoRef.current = video
    setDuration(video.duration)
    setStatus('select-person')

    await detectAtTime(0, landmarker, video)
    scrollToRef(selectPersonRef)
  }, [])

  // ── Quick automated scan alternative to manual person selection ───────────
  const preScan = useCallback(async (file) => {
    isCancelledRef.current = false
    setError(null)
    setFrames([])
    setScanFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    fileRef.current = file

    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    let landmarker
    try {
      landmarker = await createLandmarker(4)
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    const video = document.createElement('video')
    video.src = URL.createObjectURL(file)
    video.muted = true; video.playsInline = true
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror          = reject
    })

    const videoDuration = video.duration
    setDuration(videoDuration)

    const scanFps   = 1
    const frameStep = 1 / scanFps
    const totalFrames = Math.ceil(videoDuration * scanFps)

    setStatus('prescanning')

    // Reset synthetic timestamp for this scan pass
    syntheticTimestampRef.current = 0

    const results  = []
    let frameIndex = 0

    for (let t = 0; t < videoDuration; t += frameStep) {
      // Exit loop if user hits cancel
      if (isCancelledRef.current) {
        URL.revokeObjectURL(video.src)
        return
      }

      await seekToFrame(video, t)

      // ── Canvas snapshot before detection ─────────────────────────────
      // Snapshot the frame to canvas before running MediaPipe. This avoids
      // the decoder serving a different frame than the one we seeked to,
      // which causes MediaPipe to detect stale poses.
      const snapshot = snapshotVideoFrame(video)
      syntheticTimestampRef.current += Math.round(1000 / scanFps)
      const result = landmarker.detectForVideo(snapshot, syntheticTimestampRef.current)

      const thumbCanvas    = document.createElement('canvas')
      thumbCanvas.width    = 160; thumbCanvas.height = 90
      const tCtx           = thumbCanvas.getContext('2d')
      tCtx.drawImage(video, 0, 0, 160, 90)
      const thumbnail      = thumbCanvas.toDataURL('image/jpeg', 0.6)

      const persons = result.landmarks
        .map(normaliseLandmarks)
        .filter((lms) => avgConfidence(lms) >= 0.4)

      results.push({ frameIndex, timeMs: Math.round(t * 1000), thumbnail, persons })

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 100))
      if (frameIndex % 5 === 0) await new Promise((r) => setTimeout(r, 0))
    }

    // Exit if user cancels
    if (isCancelledRef.current) return

    URL.revokeObjectURL(video.src)
    setScanFrames(results)

    const maxSeen = results.reduce((max, f) => Math.max(max, f.persons.length), 0)

    if (maxSeen <= 1) {
      processVideo(null, DEFAULT_SETTINGS)
    } else {
      setStatus('select-person')
      scrollToRef(selectPersonRef)
    }
  }, [])

  // ── Detect at a specific timestamp (called by the scrubber on drag) ───────
  const detectAtTime = useCallback(async (timeS, landmarker, video) => {
    const lm  = landmarker ?? landmarkerRef.current
    const vid = video      ?? scrubVideoRef.current
    if (!lm || !vid) return

    await seekToFrame(vid, timeS)

    // Use canvas snapshot for consistent detection, same as main processing loop
    const snapshot = snapshotVideoFrame(vid)
    syntheticTimestampRef.current += 33  // ~30fps increment for scrub
    const result  = lm.detectForVideo(snapshot, syntheticTimestampRef.current)
    const persons = result.landmarks
      .map(normaliseLandmarks)
      .filter((lms) => avgConfidence(lms) >= 0.25)

    setScrubPersons(persons)
  }, [])

  // ── Single image pose extraction ──────────────────────────────────────────
  // Works for JPG, PNG, WebP, useful for pose reference images.
  const processImage = useCallback(async (file, settings = DEFAULT_SETTINGS) => {
    isCancelledRef.current = false
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)

    const modelQuality = settings.modelQuality ?? 'full'

    let landmarker
    try {
      landmarker = await createLandmarker(1, modelQuality)
    } catch (e) {
      setError('Failed to load MediaPipe model.')
      setStatus('error')
      return
    }

    // Draw image to canvas so MediaPipe can read it
    const img = new Image()
    img.src   = URL.createObjectURL(file)
    await new Promise((resolve, reject) => {
      img.onload  = resolve
      img.onerror = reject
    })

    const canvas = document.createElement('canvas')
    canvas.width  = img.naturalWidth
    canvas.height = img.naturalHeight
    canvas.getContext('2d').drawImage(img, 0, 0)
    URL.revokeObjectURL(img.src)

    const imageLandmarker = await PoseLandmarker.createFromOptions(
      visionRef.current, {
        baseOptions: { modelAssetPath: poseModelUrl(modelQuality), delegate: 'GPU' },
        runningMode: 'IMAGE',
        numPoses:    1,
        minPoseDetectionConfidence: 0.4,
      }
    )

    const result = imageLandmarker.detect(canvas)
    imageLandmarker.close()

    if (!result.landmarks?.length) {
      setError('No person detected in image.')
      setStatus('error')
      return
    }

    let landmarks        = normaliseLandmarks(result.landmarks[0])
    let worldLandmarks   = result.worldLandmarks?.[0]
      ? normaliseLandmarks(result.worldLandmarks[0])
      : null

    const trackHands = settings.trackHands ?? true

    // ── Run HandLandmarker on the image too (when hand tracking is on) ─────
    let handData = null
    if (trackHands) {
      try {
        const imageHandLandmarker = await HandLandmarker.createFromOptions(
          visionRef.current, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
            runningMode: 'IMAGE',
            numHands:    2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence:  0.5,
          }
        )
        const handResult = imageHandLandmarker.detect(canvas)
        imageHandLandmarker.close()

        if (handResult.landmarks?.length > 0) {
          handData = buildHandData(handResult)
          console.log(`[HandLandmarker] Detected ${handResult.landmarks.length} hand(s) in image`)
        }
      } catch (e) {
        console.warn('[HandLandmarker] Failed on image, skipping finger data:', e)
      }
    }

    const singleFrame = [{
      frameIndex:        0,
      timeMs:            0,
      landmarks,
      worldLandmarks,
      handData,
    }]

    originalFramesRef.current = singleFrame.map(cloneFrameData)
    setFrames(singleFrame)
    setStats({
      frameCount:    1,
      capturedCount: 1,
      keyframeCount: 1,
      duration:      '0.00',
      captureFps:    1,
      totalSampled:  1,
      modelQuality,
    })
    setStatus('done')
  }, [])

  // ── Build structured hand data from a HandLandmarker result ───────────────
  // Packages landmarks + computed finger angles into a single object per hand.
  function buildHandData(handResult) {
    const hands = {}
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const rawLms    = handResult.landmarks[i]
      const handedness = handResult.handedness[i]?.[0]?.categoryName?.toLowerCase() ?? 'left'
      // MediaPipe handedness is from the model's perspective (mirrored), flip it
      const side      = handedness === 'left' ? 'right' : 'left'
      const lms       = normaliseHandLandmarks(rawLms)
      const fingerAngles = computeFingerAnglesFromLandmarks(rawLms)
      hands[side]     = { landmarks: lms, fingerAngles, source: 'mediapipe' }
    }
    return Object.keys(hands).length > 0 ? hands : null
  }

  // ── Full video pose extraction ─────────────────────────────────────────────
  const processVideo = useCallback(async (seed, settings = DEFAULT_SETTINGS) => {
    const file = fileRef.current
    if (!file) return

    isCancelledRef.current = false
    acquireWakeLock()   // keep the machine awake for the whole (possibly long) run
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])

    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    const { captureFps, confidenceThreshold, keyframeThreshold, maxFrames, modelQuality = 'full', trackHands = true } = settings

    let landmarker
    let handLandmarker

    try {
      landmarker = await createLandmarker(1, modelQuality)
    } catch (e) {
      releaseWakeLock()
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    // Load HandLandmarker alongside pose — failure is non-fatal, finger data is
    // simply omitted. Skipped entirely when hand tracking is off, which removes a
    // full model inference per frame.
    if (trackHands) {
      try {
        handLandmarker = await createHandLandmarker()
        console.log('[HandLandmarker] Loaded successfully')
      } catch (e) {
        console.warn('[HandLandmarker] Failed to load, finger data will be omitted:', e)
        handLandmarker = null
      }
    } else {
      handLandmarker = null
      console.log('[HandLandmarker] Hand tracking disabled — skipping')
    }

    const video = document.createElement('video')
    video.src   = URL.createObjectURL(file)
    video.muted = true; video.playsInline = true
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror          = reject
    })

    const videoDuration = video.duration
    const frameStep     = 1 / captureFps
    const totalFrames   = Math.floor(videoDuration * captureFps)

    setStatus('processing')

    // Create/reset the filter bank when processing starts, matching captureFps.
    // The One Euro Filter adapts its smoothing to the signal speed, heavy smoothing at rest, light smoothing during fast motion.
    filterBankRef.current = new LandmarkFilterBank({ freq: captureFps })

    let seedLocked = seed === null
    const captured = []
    let frameIndex = 0

    // Reset synthetic timestamp for a clean processing run
    syntheticTimestampRef.current = 0

    for (let t = 0; t < videoDuration; t += frameStep) {
      // Exit if user cancels
      if (isCancelledRef.current) {
        URL.revokeObjectURL(video.src)
        releaseWakeLock()
        return
      }

      await seekToFrame(video, t)

      // ── Canvas snapshot — the primary fix for video vs image accuracy gap ──
      // We snapshot the current video frame to an offscreen canvas before running
      // MediaPipe. Two problems this solves:
      //
      // 1. Timestamp monotonicity: MediaPipe VIDEO mode requires strictly
      //    increasing timestamps. Seeking a video does not guarantee the decoder
      //    reports a monotonically increasing currentTime — it can stall, repeat,
      //    or jump back slightly. We pass our own synthetic timestamp instead,
      //    which always increases by exactly (1000/captureFps)ms per frame.
      //
      // 2. Frame stability: drawing to canvas freezes the frame so MediaPipe reads
      //    the exact pixel data we seeked to, not whatever the decoder happens to
      //    have in its buffer. This matches what IMAGE mode does naturally, which
      //    is why single-image export is more accurate.
      const snapshot = snapshotVideoFrame(video)
      syntheticTimestampRef.current += Math.round(1000 / captureFps)

      const timestampMs = syntheticTimestampRef.current
      const result      = landmarker.detectForVideo(snapshot, timestampMs)

      if (result.landmarks.length > 0) {
        let landmarks        = normaliseLandmarks(result.landmarks[0])
        let worldLandmarks   = result.worldLandmarks?.[0]
          ? normaliseLandmarks(result.worldLandmarks[0])
          : null

        if (avgConfidence(landmarks) >= confidenceThreshold) {
          if (seed && !seedLocked) {
            const center = hipCenter(landmarks)
            if (center && dist2D(center, seed) < 0.25) {
              seedLocked = true
            } else {
              frameIndex++
              setProgress(Math.round((frameIndex / totalFrames) * 90))  // capture phase: 0-90% (dominates runtime)
              continue
            }
          }

          let handData = null

          // ── Run MediaPipe HandLandmarker every frame ──────────────────
          // Pass the same canvas snapshot so both models see the same frame.
          if (handLandmarker) {
            try {
              const handResult = handLandmarker.detectForVideo(snapshot, timestampMs)
              if (handResult.landmarks?.length > 0) {
                handData = buildHandData(handResult)
              }
            } catch (e) {
              // HandLandmarker can throw if timestamp is non-monotonic, safe to ignore
            }
          }

          // ── Pass 1 collects RAW frames ────────────────────────────────
          // One Euro filtering is deferred to pass 2 (below) so it runs over the
          // full, time-ordered series instead of inline on each surviving frame.
          captured.push({
            frameIndex,
            timeMs: Math.round(t * 1000),
            landmarks,
            worldLandmarks,
            handData,           // MediaPipe hand landmarks + computed finger angles
          })
        }
      }

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 90))  // capture phase: 0-90% (dominates runtime)
      if (frameIndex % 10 === 0) await new Promise((r) => setTimeout(r, 0))
    }

    // Exit if user cancels
    if (isCancelledRef.current) { releaseWakeLock(); return }

    URL.revokeObjectURL(video.src)

    // Clean up HandLandmarker
    if (handLandmarkerRef.current) {
      handLandmarkerRef.current.close()
      handLandmarkerRef.current = null
    }

    // ── Pass 2: temporal cleanup (70-90%) ──────────────────────────────
    // First fill occluded joints by interpolating from neighbouring confident
    // frames, then run the One Euro filter. The
    // filter is stateful and order-dependent, so it must see the full, now
    // gap-free series in chronological order.
    const gapFilled = temporalGapFill(captured, confidenceThreshold)
    const cleaned = []
    for (let i = 0; i < gapFilled.length; i++) {
      if (isCancelledRef.current) { releaseWakeLock(); return }
      cleaned.push(filterBankRef.current.filter(gapFilled[i]))
      if (i % 50 === 0) {
        setProgress(90 + Math.round((i / Math.max(1, gapFilled.length)) * 7))  // cleanup: 90-97%
        await new Promise((r) => setTimeout(r, 0))
      }
    }
    setProgress(97)

    const keyframes = []
    for (const frame of cleaned) {
      if (keyframes.length === 0) { keyframes.push(frame); continue }
      const prev = keyframes[keyframes.length - 1]
      if (poseDiff(frame.landmarks, prev.landmarks) >= keyframeThreshold) {
        keyframes.push(frame)
      }
    }

    const finalFrames = subsampleFrames(keyframes, maxFrames)

    // ── Robust bone lengths for the whole clip ─────────────────────────
    // Computed from all confident cleaned frames (more samples than the final
    // subsample) so exportBVH no longer caches limb lengths from a single,
    // possibly-mangled first frame.
    const boneLengths = computeMedianBoneLengths(cleaned, confidenceThreshold)
    if (boneLengths) {
      console.log('[BVH] Median bone lengths:', boneLengths)
    }

    // How many final frames carry MediaPipe hand/finger data, for the stats panel.
    const handDataFrames = finalFrames.filter(f => f.handData).length

    originalFramesRef.current = finalFrames.map(cloneFrameData)
    setFrames(finalFrames)
    setStats({
      frameCount:    finalFrames.length,
      capturedCount: captured.length,
      keyframeCount: keyframes.length,
      duration:      videoDuration.toFixed(2),
      captureFps,
      totalSampled:  totalFrames,
      handDataFrames,
      boneLengths,
      modelQuality,
    })
    releaseWakeLock()
    setStatus('done')
    scrollToRef(statsSummaryRef)
  }, [acquireWakeLock, releaseWakeLock])

  return {
    loadVideo,
    preScan,
    detectAtTime,
    processImage,
    processVideo,
    cancelProcessing,
    clearResults,
    applyFrameEdit,
    resetFrame,
    status,
    progress,
    frames,
    scanFrames,
    stats,
    error,
    duration,
    scrubPersons,
    fileRef,
    selectPersonRef,
    statsSummaryRef,
  }
}