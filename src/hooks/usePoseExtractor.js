import { useRef, useState, useCallback } from 'react'
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { resetIKCache } from '../utils/ikSolver.js'

// MediaPipe landmark indices for the connections that'll be drawn
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

// These are the default settings the user will be able to adjust them in the UI later
export const DEFAULT_SETTINGS = {
  captureFps:          30,   // Samples per second taken from the video
  confidenceThreshold: 0.5,  // Min avg landmark visibility to keep a frame
  keyframeThreshold:   0.04, // Min avg joint movement (0–1 normalised) to keep a frame
  maxFrames:           200,  // Hard cap on final frame count
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
    const a = landmarksA[i]
    const b = landmarksB[i]
    if (!a || !b) continue
    const dx = a.x - b.x
    const dy = a.y - b.y
    const dz = a.z - b.z
    total += Math.sqrt(dx * dx + dy * dy + dz * dz)
  }
  return total / KEY_JOINT_INDICES.length
}

function subsampleFrames(frames, maxFrames) {
  if (frames.length <= maxFrames) return frames
  const step = (frames.length - 1) / (maxFrames - 1)
  return Array.from({ length: maxFrames }, (_, i) => frames[Math.round(i * step)])
}

// Hip midpoint — used as the seed position for tracking
export function hipCenter(landmarks) {
  const l = landmarks[23]
  const r = landmarks[24]
  if (!l || !r) return null
  return { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 }
}

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function normaliseLandmarks(rawLms) {
  return rawLms.map((lm) => ({
    x: parseFloat(lm.x.toFixed(4)),
    y: parseFloat(lm.y.toFixed(4)),
    z: parseFloat(lm.z.toFixed(4)),
    v: parseFloat((lm.visibility ?? 1).toFixed(3)),
  }))
}

// Main Public Hook
export function usePoseExtractor() {
  const landmarkerRef = useRef(null)
  const visionRef = useRef(null)
  const scrubVideoRef = useRef(null)
  const fileRef = useRef(null)

  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [frames, setFrames] = useState([])
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [duration, setDuration] = useState(0)
  // Detected persons at the current scrub position
  const [scrubPersons, setScrubPersons] = useState([])

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

  // ── Load video and landmarker, enter scrub mode ────────────────────
  const loadVideo = useCallback(async (file) => {
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    fileRef.current = file

    // Clean up any previous scrub video
    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    let landmarker
    try {
      landmarker = await createLandmarker(4) // detect up to 4 people
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    const video = document.createElement('video')
    video.src = URL.createObjectURL(file)
    video.muted = true
    video.playsInline = true
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })

    scrubVideoRef.current = video
    setDuration(video.duration)
    setStatus('select-person')

    await detectAtTime(0, landmarker, video) // Run detection on frame 0 immediately so canvas isn't blank
  }, [])

  // ── Detect at a specific timestamp (called by the scrubber on drag) ───────
  const detectAtTime = useCallback(async (timeS, landmarker, video) => {
    // Allow passing refs directly (first call) or use stored refs
    const lm  = landmarker  ?? landmarkerRef.current
    const vid = video       ?? scrubVideoRef.current
    if (!lm || !vid) return

    vid.currentTime = timeS
    await new Promise((r) => { vid.onseeked = r })

    const result = lm.detectForVideo(vid, Math.round(timeS * 1000))
    const persons = result.landmarks
      .map(normaliseLandmarks)
      .filter((lms) => avgConfidence(lms) >= 0.25)

    setScrubPersons(persons)
  }, [])

  // ── Full extraction, user picked a person  ────────
  const processVideo = useCallback(async (seed, settings = DEFAULT_SETTINGS) => {
    const file = fileRef.current
    if (!file) return

    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)
    setScrubPersons([])
    resetIKCache()

    // Clean up scrub video before processing
    if (scrubVideoRef.current) {
      URL.revokeObjectURL(scrubVideoRef.current.src)
      scrubVideoRef.current = null
    }

    const { captureFps, confidenceThreshold, keyframeThreshold, maxFrames } = settings

    let landmarker

    try {
      // Single person from now on as numPoses: 1 is more reliable
      landmarker = await createLandmarker(1)
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    const video = document.createElement('video')
    video.src = URL.createObjectURL(file)
    video.muted = true
    video.playsInline = true
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })

    const duration    = video.duration
    const frameStep   = 1 / captureFps
    const totalFrames = Math.floor(duration * captureFps)

    setStatus('processing')

    let seedLocked = seed === null
    const captured = []
    let frameIndex = 0

    for (let t = 0; t < duration; t += frameStep) {
      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

      const result = landmarker.detectForVideo(video, Math.round(t * 1000))

      if (result.landmarks.length > 0) {
        const landmarks = normaliseLandmarks(result.landmarks[0])
        const worldLandmarks = result.worldLandmarks?.[0]
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
          captured.push({ frameIndex, timeMs: Math.round(t * 1000), landmarks, worldLandmarks })
        }
      }

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 100))
      if (frameIndex % 10 === 0) await new Promise((r) => setTimeout(r, 0))
    }

    URL.revokeObjectURL(video.src)

    // Keyframe filter
    const keyframes = []
    for (const frame of captured) {
      if (keyframes.length === 0) { keyframes.push(frame); continue }
      const prev = keyframes[keyframes.length - 1]
      if (poseDiff(frame.landmarks, prev.landmarks) >= keyframeThreshold) {
        keyframes.push(frame)
      }
    }

    const finalFrames = subsampleFrames(keyframes, maxFrames)
    setFrames(finalFrames)
    setStats({
      frameCount:    finalFrames.length,
      capturedCount: captured.length,
      keyframeCount: keyframes.length,
      duration:      duration.toFixed(2),
      captureFps,
      totalSampled:  totalFrames,
    })
    setStatus('done')
  }, [])

  return {
    loadVideo,
    detectAtTime,
    processVideo,
    status,
    progress,
    frames,
    stats,
    error,
    duration,
    scrubPersons,
    fileRef,
  }
}