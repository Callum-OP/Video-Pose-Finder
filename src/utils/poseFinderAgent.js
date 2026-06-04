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

// ── Person identification ─────────────────────────────────────────────────────
// Call on first frame and after shot cuts. Returns { persons: [{id, bbox_center_x, bbox_center_y, confidence}], total_count }.
export async function identifyPersons(videoEl, videoId, frameIndex) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/identify-persons', { video_id: videoId, frame_index: frameIndex, image })
}

// ── Limb completion ───────────────────────────────────────────────────────────
// Call after MediaPipe detection when landmark visibility < 0.4. Returns { corrections: [{landmark_index, estimated_x, estimated_y, confidence}] }.
export async function completeLimbs(videoEl, videoId, frameIndex, landmarks) {
  if (!BACKEND) return null
  const image = frameToBase64(videoEl)
  return post('/complete-limbs', { video_id: videoId, frame_index: frameIndex, image, landmarks })
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