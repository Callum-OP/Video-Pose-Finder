import { useRef, useState, useCallback } from 'react'
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

// MediaPipe landmark indices for the connections we'll draw
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
  13: 'l_elbow',   14: 'r_elbow',
  15: 'l_wrist',   16: 'r_wrist',
  23: 'l_hip',     24: 'r_hip',
  25: 'l_knee',    26: 'r_knee',
  27: 'l_ankle',   28: 'r_ankle',
}

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

export function usePoseExtractor() {
  const landmarkerRef = useRef(null)
  const [status, setStatus] = useState('idle') // idle | loading-model | processing | done | error
  const [progress, setProgress] = useState(0)   // 0–100
  const [frames, setFrames] = useState([])       // array of landmark arrays
  const [stats, setStats] = useState(null)       // { frameCount, duration, fps }
  const [error, setError] = useState(null)

  // Load MediaPipe model once
  async function ensureLandmarker() {
    if (landmarkerRef.current) return landmarkerRef.current
    setStatus('loading-model')

    const vision = await FilesetResolver.forVisionTasks(WASM_URL)
    const landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU', // falls back to CPU automatically
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

  const processVideo = useCallback(async (file) => {
    setError(null)
    setFrames([])
    setStats(null)
    setProgress(0)

    let landmarker
    try {
      landmarker = await ensureLandmarker()
    } catch (e) {
      setError('Failed to load MediaPipe model. Check your internet connection.')
      setStatus('error')
      return
    }

    // Create a hidden video element to scrub through
    const video = document.createElement('video')
    video.src = URL.createObjectURL(file)
    video.muted = true
    video.playsInline = true

    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = reject
    })

    const duration = video.duration
    const targetFps = 30
    const frameStep = 1 / targetFps
    const totalFrames = Math.floor(duration * targetFps)

    setStatus('processing')

    const extractedFrames = []
    let frameIndex = 0

    for (let t = 0; t < duration; t += frameStep) {
      video.currentTime = t

      await new Promise((resolve) => {
        video.onseeked = resolve
      })

      // MediaPipe needs the video element to have rendered a frame
      const result = landmarker.detectForVideo(video, Math.round(t * 1000))

      if (result.landmarks.length > 0) {
        // Each landmark: { x, y, z, visibility, presence }
        extractedFrames.push({
          frameIndex,
          timeMs: Math.round(t * 1000),
          landmarks: result.landmarks[0].map((lm) => ({
            x: parseFloat(lm.x.toFixed(4)),
            y: parseFloat(lm.y.toFixed(4)),
            z: parseFloat(lm.z.toFixed(4)),
            v: parseFloat((lm.visibility ?? 1).toFixed(3)),
          })),
        })
      }

      frameIndex++
      setProgress(Math.round((frameIndex / totalFrames) * 100))

      // Yield to browser every 10 frames to keep UI responsive
      if (frameIndex % 10 === 0) {
        await new Promise((r) => setTimeout(r, 0))
      }
    }

    URL.revokeObjectURL(video.src)

    setFrames(extractedFrames)
    setStats({
      frameCount: extractedFrames.length,
      duration: duration.toFixed(2),
      fps: targetFps,
      totalFrames,
    })
    setStatus('done')
  }, [])

  return { processVideo, status, progress, frames, stats, error }
}
