import { useRef, useState, useCallback } from 'react'
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { LandmarkFilterBank } from '../utils/oneEuroFilter'
import { getOrientation, identifyPersons, completeLimbs, saveSession } from '../utils/poseFinderAgent'

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
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

const KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

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

// Apply Gemini limb corrections to a landmarks array.
// Gemini estimates positions for hidden/occluded joints from the visual context.
// Only applies corrections where Gemini confidence > 0.5 to avoid bad guesses.
function applyLimbCorrections(landmarks, corrections) {
  if (!corrections?.length) return landmarks
  const corrected = [...landmarks]
  for (const fix of corrections) {
    if (fix.confidence > 0.5 && corrected[fix.landmark_index]) {
      corrected[fix.landmark_index] = {
        ...corrected[fix.landmark_index],
        x: fix.estimated_x,
        y: fix.estimated_y,
        v: fix.confidence,
      }
    }
  }
  return corrected
}

// Main Public Hook
export function usePoseExtractor() {
  const landmarkerRef = useRef(null)
  const visionRef = useRef(null)
  const scrubVideoRef = useRef(null)
  const fileRef = useRef(null)
  const isCancelledRef = useRef(false)
  const filterBankRef = useRef(null)

  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [frames, setFrames] = useState([])
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(0)
  // Detected persons at the current scrub position
  const [scrubPersons, setScrubPersons] = useState([])
  // Storage array for the filmstrip/pre-scan view mode
  const [scanFrames, setScanFrames] = useState([])
  // Gemini-identified persons at first frame (for multi-person selection UI)
  const [geminiPersons, setGeminiPersons] = useState([])

  const selectPersonRef = useRef(null)
  const statsSummaryRef = useRef(null)

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
      minPosePresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    })
    landmarkerRef.current = landmarker
    return landmarker
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
      video.onerror = reject
    })

    const videoDuration = video.duration
    setDuration(videoDuration)
    
    const scanFps = 1 
    const frameStep = 1 / scanFps
    const totalFrames = Math.ceil(videoDuration * scanFps)

    setStatus('prescanning')

    // ── Gemini person identification on first frame ────────────────────────
    // Ask Gemini to identify how many people are in the video and where they
    // are, giving a more accurate count and description than MediaPipe alone.
    video.currentTime = 0
    await new Promise((r) => { video.onseeked = r })
    const videoId = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()
    const geminiPeople = await identifyPersons(video, videoId, 0)
    if (geminiPeople?.persons?.length > 0) {
      setGeminiPersons(geminiPeople.persons)
      console.log(`[Gemini] Identified ${geminiPeople.total_count} person(s) in first frame`)
    }

    const results = []
    let frameIndex = 0

    for (let t = 0; t < videoDuration; t += frameStep) {
      // Exit loop if user hits cancel
      if (isCancelledRef.current) {
        URL.revokeObjectURL(video.src)
        return
      }

      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

      const result = landmarker.detectForVideo(video, Math.round(t * 1000))

      const thumbCanvas = document.createElement('canvas')
      thumbCanvas.width = 160; thumbCanvas.height = 90
      const tCtx = thumbCanvas.getContext('2d')
      tCtx.drawImage(video, 0, 0, 160, 90)
      const thumbnail = thumbCanvas.toDataURL('image/jpeg', 0.6)

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
    const lm = landmarker ?? landmarkerRef.current
    const vid = video ?? scrubVideoRef.current
    if (!lm || !vid) return

    vid.currentTime = timeS
    await new Promise((r) => { vid.onseeked = r })

    const result = lm.detectForVideo(vid, Math.round(timeS * 1000))
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
    img.src = URL.createObjectURL(file)
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = reject
    })

    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    canvas.getContext('2d').drawImage(img, 0, 0)
    URL.revokeObjectURL(img.src)

    const imageLandmarker = await PoseLandmarker.createFromOptions(
      visionRef.current, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        numPoses: 1,
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
    const worldLandmarks = result.worldLandmarks?.[0]
      ? normaliseLandmarks(result.worldLandmarks[0])
      : null

    // ── Gemini limb completion for images ─────────────────────────────────
    // Images often have partially visible people (cropped, occluded).
    // Gemini can reason about what the hidden limbs likely look like.
    const hasLowConfidence = landmarks.some(lm => (lm.v ?? 1) < 0.4)
    if (hasLowConfidence) {
      const imageId   = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()
      const completion = await completeLimbs(canvas, imageId, 0, landmarks)
      if (completion?.corrections?.length) {
        landmarks = applyLimbCorrections(landmarks, completion.corrections)
        console.log(`[Gemini] Applied ${completion.corrections.length} limb correction(s) to image`)
      }
    }

    const singleFrame = [{
      frameIndex:     0,
      timeMs:         0,
      landmarks,
      worldLandmarks,
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

    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    const { captureFps, confidenceThreshold, keyframeThreshold, maxFrames } = settings

    let landmarker

    try {
      landmarker = await createLandmarker(1)
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

    const videoDuration = video.duration
    const frameStep = 1 / captureFps
    const totalFrames = Math.floor(videoDuration * captureFps)

    setStatus('processing')

    // Create/reset the filter bank when processing starts, matching captureFps.
    // The One Euro Filter adapts its smoothing to the signal speed, heavy smoothing at rest, light smoothing during fast motion.
    filterBankRef.current = new LandmarkFilterBank({ freq: captureFps })

    // Stable video ID derived from filename, used as MongoDB cache key so
    // re-processing the same video reuses Gemini's previous orientation answers.
    const videoId = file.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()

    // Gemini orientation: sample at 1fps regardless of captureFps.
    // Orientation changes slowly enough that 1fps is sufficient, and this avoids hammering the Gemini API on every frame.
    let lastOrientationTime    = -1
    const ORIENTATION_INTERVAL = 1  // Seconds between Gemini orientation calls

    // Gemini limb completion: throttled to at most once per 2 seconds.
    // Limb completion calls are expensive so only trigger when needed and not too frequently so processing stays fast.
    let lastLimbCompletionTime    = -2
    const LIMB_COMPLETION_INTERVAL = 2  // Seconds

    let seedLocked = seed === null
    const captured = []
    let frameIndex = 0

    for (let t = 0; t < videoDuration; t += frameStep) {
      // Exit if user cancels
      if (isCancelledRef.current) {
        URL.revokeObjectURL(video.src)
        return
      }

      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

      const result = landmarker.detectForVideo(video, Math.round(t * 1000))

      if (result.landmarks.length > 0) {
        let landmarks        = normaliseLandmarks(result.landmarks[0])
        const worldLandmarks = result.worldLandmarks?.[0] ? normaliseLandmarks(result.worldLandmarks[0]) : null

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

          // ── Gemini orientation (1fps) ─────────────────────────────────
          // Ask Gemini which way the person is facing. 
          // The answer gets used by OrientationEstimator in exportBVH to rotate the hips bone correctly.
          let geminiOrientation = null
          if (t - lastOrientationTime >= ORIENTATION_INTERVAL) {
            lastOrientationTime = t
            geminiOrientation = await getOrientation(video, videoId, frameIndex)
          }

          // ── Gemini limb completion (throttled, when landmarks are hidden) ─
          // When MediaPipe can't see a joint (visibility < 0.4, e.g. behind another person or off-frame), 
          // Gemini estimates its position from the visual context and the visible anatomy.
          const hasLowConfidence = landmarks.some(lm => (lm.v ?? 1) < 0.4)
          if (hasLowConfidence && t - lastLimbCompletionTime >= LIMB_COMPLETION_INTERVAL) {
            lastLimbCompletionTime = t
            const completion = await completeLimbs(video, videoId, frameIndex, landmarks)
            if (completion?.corrections?.length) {
              landmarks = applyLimbCorrections(landmarks, completion.corrections)
            }
          }

          // ── Filter + store frame ───────────────────────────────────────
          // One Euro Filter runs on the (already Gemini-corrected) landmarks.
          // geminiOrientation is attached so exportBVH's OrientationEstimator can use it.
          const filteredFrame = filterBankRef.current.filter({
            frameIndex,
            timeMs: Math.round(t * 1000),
            landmarks,
            worldLandmarks,
            geminiOrientation,
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

    const keyframes = []
    for (const frame of captured) {
      if (keyframes.length === 0) { keyframes.push(frame); continue }
      const prev = keyframes[keyframes.length - 1]
      if (poseDiff(frame.landmarks, prev.landmarks) >= keyframeThreshold) {
        keyframes.push(frame)
      }
    }

    const finalFrames = subsampleFrames(keyframes, maxFrames)

    // Compute orientation summary for the stats panel
    const viewCounts = finalFrames.reduce((acc, f) => {
      const v = f.orientation?.view ?? 'unknown'
      acc[v] = (acc[v] || 0) + 1
      return acc
    }, {})
    const shotCuts = finalFrames.filter(f => f.orientation?.shotCut).length

    // Count how many frames had Gemini orientation data (vs heuristic fallback)
    const geminiOrientationFrames = finalFrames.filter(f => f.geminiOrientation).length

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
    })
    setStatus('done')
    scrollToRef(statsSummaryRef)

    // ── Save session to MongoDB ──────────────────────────────────────────────
    // Persists the session so the user can retrieve it later without reprocessing.
    saveSession(videoId, finalFrames.length, {
      duration:      videoDuration.toFixed(2),
      captureFps,
      frameCount:    finalFrames.length,
      shotCuts,
      viewCounts,
      geminiOrientationFrames,
    })
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
    fileRef,
    selectPersonRef,
    statsSummaryRef,
  }
}