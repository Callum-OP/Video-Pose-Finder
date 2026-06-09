// ── PoseFinder Agent ──────────────────────────────────────────────────
// Connects the frontend to the PoseFinder backend agent on Google Cloud.
// If no backend URL is configured, then it will silently return null and the existing pipeline runs unchanged.

const BACKEND = import.meta.env.VITE_BACKEND_URL || null

// Convert a video element's current frame to a base64 JPEG
function frameToBase64(videoOrCanvas, quality = 0.8) {
  const canvas = document.createElement('canvas')
  canvas.width  = videoOrCanvas.videoWidth  || videoOrCanvas.width
  canvas.height = videoOrCanvas.videoHeight || videoOrCanvas.height
  canvas.getContext('2d').drawImage(videoOrCanvas, 0, 0)
  const dataUrl = canvas.toDataURL('image/jpeg', quality)
  return dataUrl.split(',')[1]  // Strip the data:image/jpeg;base64, prefix.
}

// ── Snapshot a video frame to an offscreen canvas ─────────────────────────────
// This is the key fix for video vs image accuracy. Instead of passing the video
// element directly to MediaPipe (which reads whatever frame the decoder has),
// we draw the current frame to a canvas first. The canvas is a stable snapshot
// that won't change mid-detection, matching what the image path does.
export function snapshotVideoFrame(videoEl) {
  const canvas = document.createElement('canvas')
  canvas.width  = videoEl.videoWidth
  canvas.height = videoEl.videoHeight
  canvas.getContext('2d').drawImage(videoEl, 0, 0)
  return canvas
}

// ── Crop the frame to the tracked person before sending to Gemini ─────────────
// Prevents Gemini guessing positions for the wrong person in multi-person scenes.
// Returns base64 of the cropped region, plus the crop bounds so corrections can
// be un-normalised back to full-frame coordinates afterward.
function cropPersonToBase64(videoOrCanvas, landmarks, quality = 0.8) {
  const srcW = videoOrCanvas.videoWidth  || videoOrCanvas.width
  const srcH = videoOrCanvas.videoHeight || videoOrCanvas.height

  const xs = landmarks.map(l => l.x)
  const ys = landmarks.map(l => l.y)
  const pad = 0.06
  const minX = Math.max(0, Math.min(...xs) - pad)
  const minY = Math.max(0, Math.min(...ys) - pad)
  const maxX = Math.min(1, Math.max(...xs) + pad)
  const maxY = Math.min(1, Math.max(...ys) + pad)

  const cropW = Math.round((maxX - minX) * srcW)
  const cropH = Math.round((maxY - minY) * srcH)

  const crop = document.createElement('canvas')
  crop.width  = cropW
  crop.height = cropH
  crop.getContext('2d').drawImage(
    videoOrCanvas,
    minX * srcW, minY * srcH, cropW, cropH,
    0, 0, cropW, cropH
  )

  // Re-normalise landmarks to crop-relative coordinates for the prompt
  const croppedLandmarks = landmarks.map(lm => ({
    ...lm,
    x: (lm.x - minX) / (maxX - minX),
    y: (lm.y - minY) / (maxY - minY),
  }))

  return {
    image:            crop.toDataURL('image/jpeg', quality).split(',')[1],
    croppedLandmarks,
    cropBounds:       { minX, minY, maxX, maxY },
  }
}

// After /complete-limbs returns corrections in crop space, convert back to full-frame space.
function unNormCorrections(corrections, cropBounds) {
  if (!corrections?.length || !cropBounds) return corrections
  const { minX, minY, maxX, maxY } = cropBounds
  return corrections.map(fix => ({
    ...fix,
    estimated_x: minX + fix.estimated_x * (maxX - minX),
    estimated_y: minY + fix.estimated_y * (maxY - minY),
    // z_depth is already in world-relative units, no un-normalisation needed
  }))
}

async function post(endpoint, body) {
  if (!BACKEND) return null
  try {
    const res = await fetch(`${BACKEND}${endpoint}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null  // Backend unavailable
  }
}

// ── Orientation oracle ────────────────────────────────────────────────────────
// Call at 1fps during processing. Returns { yaw_degrees, view, confidence } or null if backend unavailable.
export async function getOrientation(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/orientation', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Gravity/lying-down orientation ───────────────────────────────────────────
// Call at 1fps alongside /orientation. Returns how tilted the person's spine is
// relative to gravity — catches lying down, falling, floor poses that MediaPipe
// cannot handle because it assumes an upright skeleton.
export async function getGravityOrientation(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/gravity-orientation', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Person identification ─────────────────────────────────────────────────────
// Call on first frame and after shot cuts. Returns { persons: [{id, bbox_center_x, bbox_center_y, confidence}], total_count }.
export async function identifyPersons(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/identify-persons', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Limb completion — crops to the person first ───────────────────────────────
// Call after MediaPipe detection when landmark visibility < 0.4.
// Returns { corrections: [{landmark_index, estimated_x, estimated_y, z_depth, confidence}] }.
// z_depth is now included so world-landmark Z can be corrected too.
export async function completeLimbs(videoEl, videoId, frameIndex, landmarks) {
  if (!BACKEND) return null

  const { image, croppedLandmarks, cropBounds } = cropPersonToBase64(videoEl, landmarks)

  const result = await post('/complete-limbs', {
    video_id:    videoId,
    frame_index: frameIndex,
    image,
    landmarks:   croppedLandmarks,  // crop-relative coords
  })

  if (!result?.corrections?.length) return result

  // Convert corrections back to full-frame coordinates before returning
  return { ...result, corrections: unNormCorrections(result.corrections, cropBounds) }
}

// ── Hand pose detection ───────────────────────────────────────────────────────
// Call at 1fps alongside orientation. Returns { hands: [{side, pose, wrist_rotation, confidence}] }.
// Used as a fallback when MediaPipe HandLandmarker fails to detect (hands too small/occluded).
export async function getHandPoses(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/hand-poses', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Finger angle estimation (Gemini fallback) ─────────────────────────────────
// Call when MediaPipe HandLandmarker doesn't detect hands.
// Returns per-joint bend angles (0=straight, 90=fully bent) for each finger.
export async function getFingerPoses(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/finger-poses', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Clip naming ───────────────────────────────────────────────────────────────
// Call once after processing finishes, before BVH export. Samples up to 4 evenly
// spaced frames from the captured frames array and asks Gemini what movement it sees.
// Returns { filename, activity } — filename used as the BVH download name.
export async function nameClip(frames, videoId) {
  if (!BACKEND || !frames?.length) return null

  // Sample up to 4 evenly spaced frames to give Gemini a view of the motion arc
  const indices = frames.length <= 4
    ? frames.map((_, i) => i)
    : [0, 1, 2, 3].map(i => Math.round(i * (frames.length - 1) / 3))

  // Each frame needs a canvas snapshot to convert to JPEG
  // frames carry worldLandmarks but no canvas — we'll pass null images the backend handles gracefully
  // The caller (usePoseExtractor) passes the video element so we can snapshot it;
  // but at export time the video is gone. Instead we accept pre-rendered canvases or skip.
  // See usage in usePoseExtractor: nameClip(frames, videoEl, videoId)
  const images = indices
    .map(i => frames[i]?._snapshotB64)
    .filter(Boolean)

  if (!images.length) return null

  return post('/name-clip', { video_id: videoId, images })
}

// ── Session persistence ───────────────────────────────────────────────────────
export async function saveSession(videoId, frameCount, stats) {
  return post('/session/save', { video_id: videoId, frame_count: frameCount, stats })
}

export async function getSessionHistory() {
  if (!BACKEND) return []
  try {
    const res = await fetch(`${BACKEND}/session/history`)
    if (!res.ok) return []
    const data = await res.json()
    return data.sessions || []
  } catch {
    return []
  }
}