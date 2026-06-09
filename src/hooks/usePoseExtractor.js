import { useRef, useState, useCallback } from 'react'
import { PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { LandmarkFilterBank } from '../utils/oneEuroFilter'
import { getOrientation, getGravityOrientation, identifyPersons, completeLimbs, saveSession, getHandPoses, getFingerPoses, nameClip, snapshotVideoFrame } from '../utils/poseFinderAgent'

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
  captureFps:          30,   // Samples per second taken from the video.
  confidenceThreshold: 0.5,  // Min accepted amount of visibility to keep a frame/pose.
  keyframeThreshold:   0.04, // Min accepted amount of joint movement (0–1 normalised) to keep a frame/pose.
  maxFrames:           200,  // The number of frames/poses to keep.
}

export const PERSON_COLORS = ['#7c6cff', '#39e8a0', '#f5a623', '#ff4d6d']

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
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

// Apply Gemini limb corrections to a landmarks array.
// Gemini estimates positions for hidden/occluded joints from the visual context.
// Only applies corrections where Gemini confidence > 0.5 to avoid bad guesses.
function applyLimbCorrections(landmarks, worldLandmarks, corrections) {
  if (!corrections?.length) return { landmarks, worldLandmarks }

  const correctedLms   = [...landmarks]
  const correctedWorld = worldLandmarks ? [...worldLandmarks] : null

  for (const fix of corrections) {
    if (fix.confidence > 0.5 && correctedLms[fix.landmark_index]) {
      // Apply 2D image-space correction to screen landmarks
      correctedLms[fix.landmark_index] = {
        ...correctedLms[fix.landmark_index],
        x: fix.estimated_x,
        y: fix.estimated_y,
        v: fix.confidence,
      }

      // Apply Z depth correction to world landmarks if available.
      // This is the key improvement over the old version — previously only X/Y were corrected,
      // leaving the 3D skeleton depth wrong for occluded joints. Gemini now estimates
      // depth from body proportions and foreshortening cues in the image.
      if (correctedWorld?.[fix.landmark_index] && fix.z_depth !== undefined) {
        correctedWorld[fix.landmark_index] = {
          ...correctedWorld[fix.landmark_index],
          z: fix.z_depth,
          v: fix.confidence,
        }
      }
    }
  }

  return { landmarks: correctedLms, worldLandmarks: correctedWorld }
}

// ── Compute finger bend angles from MediaPipe hand landmarks ──────────────────
// For each finger, compute bend at each joint using the angle between consecutive bone vectors.
// Returns a structure matching the Gemini finger-poses format so BVH export can consume either source.
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

// Main Public Hook
export function usePoseExtractor() {
  const landmarkerRef     = useRef(null)
  const handLandmarkerRef = useRef(null)   // MediaPipe HandLandmarker
  const visionRef         = useRef(null)
  const scrubVideoRef     = useRef(null)
  const fileRef           = useRef(null)
  const isCancelledRef    = useRef(false)
  const filterBankRef     = useRef(null)

  // ── Monotonic timestamp counter ───────────────────────────────────────────
  // This is the primary fix for why video is less accurate than images.
  // MediaPipe's VIDEO mode requires strictly increasing timestamps. When seeking
  // through a video frame-by-frame, the decoder's reported time can be
  // non-monotonic or stall at the same value, causing MediaPipe to return stale
  // results from the previous frame. We maintain our own counter that always
  // increases, decoupled from actual video time.
  const syntheticTimestampRef = useRef(0)

  const [status, setStatus]           = useState('idle')
  const [progress, setProgress]       = useState(0)
  const [frames, setFrames]           = useState([])
  const [stats, setStats]             = useState(null)
  const [error, setError]             = useState(null)
  const [duration, setDuration]       = useState(0)
  // Detected persons at the current scrub position
  const [scrubPersons, setScrubPersons]   = useState([])
  // Storage array for the filmstrip/pre-scan view mode
  const [scanFrames, setScanFrames]       = useState([])
  // Gemini-identified persons at first frame (for multi-person selection UI)
  const [geminiPersons, setGeminiPersons] = useState([])
  // Clip name returned by Gemini after processing, used as BVH filename
  const [clipName, setClipName]           = useState(null)

  const selectPersonRef  = useRef(null)
  const statsSummaryRef  = useRef(null)

  const scrollToRef = (targetRef) => {
    setTimeout(() => {
      if (targetRef.current) {
        targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 150)
  }

  const cancelProcessing = useCallback(() => {
    isCancelledRef.current = true
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    setScanFrames([])
    setGeminiPersons([])
    setClipName(null)
    fileRef.current = null
    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }
    setStatus('idle')
  }, [])

  async function createLandmarker(numPoses) {
    if (!visionRef.current) {
      setStatus('loading-model')
      visionRef.current = await FilesetResolver.forVisionTasks(WASM_URL)
    }
    if (landmarkerRef.current) {
      landmarkerRef.current.close()
      landmarkerRef.current = null
    }
    const landmarker = await PoseLandmarker.createFromOptions(visionRef.current, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
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
    setGeminiPersons([])
    setClipName(null)
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
    setGeminiPersons([])
    setClipName(null)
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

    // Read toggle preference right before execution loop
    const isLocalMode = localStorage.getItem('use_local_backend') === 'true'

    if (!isLocalMode) {
      video.currentTime = 0
      await new Promise((r) => { video.onseeked = r })
      const videoId      = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()
      const geminiPeople = await identifyPersons(video, videoId, 0)
      if (geminiPeople?.persons?.length > 0) {
        setGeminiPersons(geminiPeople.persons)
        console.log(`[Gemini] Identified ${geminiPeople.total_count} person(s) in first frame`)
      }
    } else {
      console.log('[Pipeline] Local Mode Active: Skipping Gemini identification pass.')
    }

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

      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

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

    vid.currentTime = timeS
    await new Promise((r) => { vid.onseeked = r })

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
  // Also runs Gemini limb completion on the result for better accuracy.
  // Works for JPG, PNG, WebP, useful for pose reference images.
  const processImage = useCallback(async (file) => {
    isCancelledRef.current = false
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setClipName(null)

    let landmarker
    try {
      landmarker = await createLandmarker(1)
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
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
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

    // ── Run HandLandmarker on the image too ───────────────────────────────
    let handData = null
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

    const isLocalMode       = localStorage.getItem('use_local_backend') === 'true'
    const hasLowConfidence  = landmarks.some(lm => (lm.v ?? 1) < 0.4)

    if (hasLowConfidence && !isLocalMode) {
      const imageId    = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()
      const completion = await completeLimbs(canvas, imageId, 0, landmarks)
      if (completion?.corrections?.length) {
        // Apply corrections to both 2D landmarks and world landmarks (including Z depth)
        const corrected = applyLimbCorrections(landmarks, worldLandmarks, completion.corrections)
        landmarks     = corrected.landmarks
        worldLandmarks = corrected.worldLandmarks
        console.log(`[Gemini] Applied ${completion.corrections.length} limb correction(s) to image`)
      }
    }

    // If MediaPipe hands didn't detect and backend is available, ask Gemini
    let geminiFingerPoses = null
    if (!handData && !isLocalMode) {
      const imageId = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()
      geminiFingerPoses = await getFingerPoses(canvas, imageId, 0)
    }

    const singleFrame = [{
      frameIndex:        0,
      timeMs:            0,
      landmarks,
      worldLandmarks,
      handData,
      geminiFingerPoses,
    }]

    setFrames(singleFrame)
    setStats({
      frameCount:    1,
      capturedCount: 1,
      keyframeCount: 1,
      duration:      '0.00',
      captureFps:    1,
      totalSampled:  1,
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
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    setClipName(null)

    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    const { captureFps, confidenceThreshold, keyframeThreshold, maxFrames } = settings

    let landmarker
    let handLandmarker

    try {
      landmarker = await createLandmarker(1)
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    // Load HandLandmarker alongside pose — failure is non-fatal, finger data becomes Gemini-only
    try {
      handLandmarker = await createHandLandmarker()
      console.log('[HandLandmarker] Loaded successfully')
    } catch (e) {
      console.warn('[HandLandmarker] Failed to load, will use Gemini fallback for fingers:', e)
      handLandmarker = null
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

    // Stable video ID derived from filename, used as MongoDB cache key so
    // re-processing the same video reuses Gemini's previous orientation answers.
    const videoId = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()

    let lastOrientationTime       = -1
    const ORIENTATION_INTERVAL    = 1   // seconds

    // Gemini limb completion: trigger on every frame with new occlusions.
    // Previously this had a 3-second minimum interval which meant occlusions
    // during fast movement went uncorrected for too long.
    let lastLimbCompletionTime    = -2
    const LIMB_COMPLETION_INTERVAL = 1  // seconds — tighter interval for better tracking

    // Track which joints were occluded last frame to detect transition into occlusion
    let prevOccludedJoints = new Set()

    // Gravity orientation state — persisted across frames, updated at 1fps
    // alongside /orientation. Non-standing poses need the BVH skeleton rotated.
    let currentGravityOrientation = null

    let seedLocked = seed === null
    const captured = []
    let frameIndex = 0

    // Reset synthetic timestamp for a clean processing run
    syntheticTimestampRef.current = 0

    // Capture the static local mode check parameter
    const isLocalMode = localStorage.getItem('use_local_backend') === 'true'

    for (let t = 0; t < videoDuration; t += frameStep) {
      // Exit if user cancels
      if (isCancelledRef.current) {
        URL.revokeObjectURL(video.src)
        return
      }

      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

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
              setProgress(Math.round((frameIndex / totalFrames) * 100))
              continue
            }
          }

          let geminiOrientation     = null
          let geminiHandPoses       = null
          let geminiFingerPoses     = null
          let handData              = null

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

          if (!isLocalMode) {
            // ── Orientation + hand poses + gravity at 1fps ────────────
            if (t - lastOrientationTime >= ORIENTATION_INTERVAL) {
              lastOrientationTime = t

              // Run orientation, gravity, and hand poses in parallel for speed
              const [orientResult, gravityResult, handPoseResult] = await Promise.all([
                getOrientation(video, videoId, frameIndex),
                getGravityOrientation(video, videoId, frameIndex),
                getHandPoses(video, videoId, frameIndex),
              ])

              geminiOrientation         = orientResult
              currentGravityOrientation = gravityResult  // Persist across frames until next update
              geminiHandPoses           = handPoseResult

              // Only ask Gemini for fingers if MediaPipe didn't detect hands this second
              if (!handData) {
                geminiFingerPoses = await getFingerPoses(video, videoId, frameIndex)
                if (geminiFingerPoses) {
                  console.log(`[Gemini] Finger fallback used at t=${t.toFixed(2)}s`)
                }
              }
            }

            // ── Limb completion: trigger on new or ongoing occlusions ─────
            // Now fires every LIMB_COMPLETION_INTERVAL seconds (down from 3s)
            // whenever any joint is occluded. Previously ongoing occlusions
            // went uncorrected between interval triggers.
            const currentOccluded = new Set(
              landmarks
                .map((lm, i) => ({ i, v: lm.v ?? 1 }))
                .filter(({ v }) => v < 0.4)
                .map(({ i }) => i)
            )
            const newlyOccluded = [...currentOccluded].filter(i => !prevOccludedJoints.has(i))
            const hasOcclusion  = currentOccluded.size > 0

            if (hasOcclusion && (newlyOccluded.length > 0 || t - lastLimbCompletionTime >= LIMB_COMPLETION_INTERVAL)) {
              lastLimbCompletionTime = t
              const completion = await completeLimbs(snapshot, videoId, frameIndex, landmarks)
              if (completion?.corrections?.length) {
                // Apply to both screen landmarks and world landmarks (with Z depth)
                const corrected = applyLimbCorrections(landmarks, worldLandmarks, completion.corrections)
                landmarks     = corrected.landmarks
                worldLandmarks = corrected.worldLandmarks
                console.log(`[Gemini] ${newlyOccluded.length > 0 ? 'New' : 'Ongoing'} occlusion — applied ${completion.corrections.length} correction(s) at t=${t.toFixed(2)}s`)
              }
            }

            prevOccludedJoints = currentOccluded

          } else {
            // ── Local Fallback Mode Path ──────────────────────────────
            if (t - lastOrientationTime >= ORIENTATION_INTERVAL) {
              lastOrientationTime = t
              geminiOrientation   = { source: 'local_fallback', yaw: 0, view: 'front', shotCut: false }
            }
          }

          // ── Snapshot base64 for clip naming ───────────────────────────
          // Store a JPEG snapshot on a small subset of frames so nameClip()
          // can sample them after processing completes without needing the video.
          // Only store on ~4 evenly spaced frames to avoid memory pressure.
          let _snapshotB64 = undefined
          const isNamingFrame = frameIndex === 0
            || frameIndex === Math.floor(totalFrames * 0.33)
            || frameIndex === Math.floor(totalFrames * 0.66)
            || frameIndex === totalFrames - 1
          if (isNamingFrame && !isLocalMode) {
            _snapshotB64 = snapshot.toDataURL('image/jpeg', 0.6).split(',')[1]
          }

          const filteredFrame = filterBankRef.current.filter({
            frameIndex,
            timeMs: Math.round(t * 1000),
            landmarks,
            worldLandmarks,
            geminiOrientation,
            geminiHandPoses,
            geminiFingerPoses,
            geminiGravityOrientation: currentGravityOrientation,  // Passed through to BVH exporter
            handData,           // MediaPipe hand landmarks + computed finger angles
            _snapshotB64,       // Retained for clip naming only, not exported to BVH
          })
          captured.push(filteredFrame)
        }
      }

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 100))
      if (frameIndex % 10 === 0) await new Promise((r) => setTimeout(r, 0))
    }

    // Exit if user cancels
    if (isCancelledRef.current) return

    URL.revokeObjectURL(video.src)

    // Clean up HandLandmarker
    if (handLandmarkerRef.current) {
      handLandmarkerRef.current.close()
      handLandmarkerRef.current = null
    }

    const keyframes = []
    for (const frame of captured) {
      if (keyframes.length === 0) { keyframes.push(frame); continue }
      const prev = keyframes[keyframes.length - 1]
      if (poseDiff(frame.landmarks, prev.landmarks) >= keyframeThreshold) {
        keyframes.push(frame)
      }
    }

    const finalFrames = subsampleFrames(keyframes, maxFrames)

    // ── Clip naming via Gemini ─────────────────────────────────────────
    // Samples the stored JPEG snapshots from the processed frames and asks
    // Gemini what movement it sees. The result becomes the BVH filename.
    let resolvedClipName = null
    if (!isLocalMode) {
      try {
        const namingResult = await nameClip(finalFrames, videoId)
        if (namingResult?.filename) {
          resolvedClipName = namingResult.filename
          setClipName(namingResult)
          console.log(`[Gemini] Clip named: "${namingResult.filename}" (${namingResult.activity})`)
        }
      } catch (e) {
        console.warn('[Gemini] Clip naming failed, using default filename:', e)
      }
    }

    // Compute orientation summary for the stats panel.
    const viewCounts = finalFrames.reduce((acc, f) => {
      const v = f.orientation?.view ?? 'unknown'
      acc[v]  = (acc[v] || 0) + 1
      return acc
    }, {})
    const shotCuts                = finalFrames.filter(f => f.orientation?.shotCut).length
    const geminiOrientationFrames = finalFrames.filter(f => f.geminiOrientation).length
    const handDataFrames          = finalFrames.filter(f => f.handData).length
    const geminiFingerFrames      = finalFrames.filter(f => f.geminiFingerPoses).length
    const gravityFrames           = finalFrames.filter(f => f.geminiGravityOrientation?.confidence > 0.5).length
    const poseTypes               = [...new Set(finalFrames.map(f => f.geminiGravityOrientation?.pose_type).filter(Boolean))]

    setFrames(finalFrames)
    setStats({
      frameCount:    finalFrames.length,
      capturedCount: captured.length,
      keyframeCount: keyframes.length,
      duration:      videoDuration.toFixed(2),
      captureFps,
      totalSampled:  totalFrames,
      viewCounts,
      shotCuts,
      geminiOrientationFrames,
      handDataFrames,
      geminiFingerFrames,
      gravityFrames,
      poseTypes,
      clipName:      resolvedClipName,
    })
    setStatus('done')
    scrollToRef(statsSummaryRef)

    // Save session database call logic.
    if (!isLocalMode) {
      saveSession(videoId, finalFrames.length, {
        duration:      videoDuration.toFixed(2),
        captureFps,
        frameCount:    finalFrames.length,
        shotCuts,
        viewCounts,
        geminiOrientationFrames,
        handDataFrames,
        geminiFingerFrames,
        gravityFrames,
        poseTypes,
        clipName:      resolvedClipName,
      })
    } else {
      console.log('[Pipeline] Local Mode Active: Skipping MongoDB cloud save session synchronization.')
    }
  }, [])

  return {
    loadVideo,
    preScan,
    detectAtTime,
    processImage,
    processVideo,
    cancelProcessing,
    status,
    progress,
    frames,
    scanFrames,
    stats,
    error,
    duration,
    scrubPersons,
    geminiPersons,
    clipName,
    fileRef,
    selectPersonRef,
    statsSummaryRef,
  }
}