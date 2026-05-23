import { useRef, useState, useCallback } from 'react'
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

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

// Named joints for the data panel
export const NAMED_JOINTS = {
  0:  'nose',
  11: 'l_shoulder', 12: 'r_shoulder',
  13: 'l_elbow',    14: 'r_elbow',
  15: 'l_wrist',    16: 'r_wrist',
  23: 'l_hip',      24: 'r_hip',
  25: 'l_knee',     26: 'r_knee',
  27: 'l_ankle',    28: 'r_ankle',
}

// Default extraction settings — exported so App.jsx can initialise its state
export const DEFAULT_SETTINGS = {
  captureFps:          12,   // Samples per second taken from the video
  confidenceThreshold: 0.5,  // Min avg landmark visibility to keep a frame
  keyframeThreshold:   0.04, // Min avg joint movement (0–1 normalised) to keep a frame
  maxFrames:           60,   // Hard cap on final frame count
}

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

// ── Helpers ──────────────────────────────────────────────────────────────────

// Average visibility across the 12 key body joints (ignores face noise)
const KEY_JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

function avgConfidence(landmarks) {
  const sum = KEY_JOINT_INDICES.reduce((acc, i) => acc + (landmarks[i]?.v ?? 0), 0)
  return sum / KEY_JOINT_INDICES.length
}

// Mean euclidean distance between key joints across two frames
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

// After keyframe filtering, if it still exceeds maxFrames, evenly subsample
function subsampleFrames(frames, maxFrames) {
  if (frames.length <= maxFrames) return frames
  const step = (frames.length - 1) / (maxFrames - 1)
  return Array.from({ length: maxFrames }, (_, i) => frames[Math.round(i * step)])
}

export function usePoseExtractor() {
  const landmarkerRef = useRef(null)
  const visionRef = useRef(null)
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [frames, setFrames] = useState([])
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  async function createLandmarker() {
    // Load the WASM fileset once and cache it
    if (!visionRef.current) {
      setStatus('loading-model')
      visionRef.current = await FilesetResolver.forVisionTasks(WASM_URL)
    }

    // Always create a fresh landmarker so its internal timestamp counter resets
    if (landmarkerRef.current) {
      landmarkerRef.current.close()
      landmarkerRef.current = null
    }

    const landmarker = await PoseLandmarker.createFromOptions(visionRef.current, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })

    landmarkerRef.current = landmarker
    return landmarker
  }

  // Settings shape matches DEFAULT_SETTINGS
  const processVideo = useCallback(async (file, settings = DEFAULT_SETTINGS) => {
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)

    const { captureFps, confidenceThreshold, keyframeThreshold, maxFrames } = settings

    let landmarker
    try {
      landmarker = await createLandmarker()
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

    // Capture at a set fps and use a confidence filter
    const captured = []
    let frameIndex = 0

    for (let t = 0; t < duration; t += frameStep) {
      video.currentTime = t
      await new Promise((r) => { video.onseeked = r })

      const result = landmarker.detectForVideo(video, Math.round(t * 1000))

      if (result.landmarks.length > 0) {
        const landmarks = result.landmarks[0].map((lm) => ({
          x: parseFloat(lm.x.toFixed(4)),
          y: parseFloat(lm.y.toFixed(4)),
          z: parseFloat(lm.z.toFixed(4)),
          v: parseFloat((lm.visibility ?? 1).toFixed(3)),
        }))

        if (avgConfidence(landmarks) >= confidenceThreshold) {
          captured.push({ frameIndex, timeMs: Math.round(t * 1000), landmarks })
        }
      }

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 100))
      if (frameIndex % 10 === 0) await new Promise((r) => setTimeout(r, 0))
    }

    URL.revokeObjectURL(video.src)

    // Keyframe filter, cull frames too similar to previous
    const keyframes = []
    for (const frame of captured) {
      if (keyframes.length === 0) {
        keyframes.push(frame)
        continue
      }
      const prev = keyframes[keyframes.length - 1]
      if (poseDiff(frame.landmarks, prev.landmarks) >= keyframeThreshold) {
        keyframes.push(frame)
      }
    }

    // Max frames cap, limit total number of frames
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

  return { processVideo, status, progress, frames, stats, error }
}