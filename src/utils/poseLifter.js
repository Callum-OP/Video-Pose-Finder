// ── Pose Lifter Enhancement ─────────────────────────────────────────────────────────────
// Loads MHFormer ONNX model from public/models/ and lifts MediaPipe 2D landmarks to 3D.

import * as ort from 'onnxruntime-web'

// ── MediaPipe 33 to H36M 17 joint mapping ─────────────────────────────────────
// MHFormer expects exactly 17 joints in H36M order.
// We derive them from MediaPipe's 33 landmarks.
// H36M order: Hip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
//             Spine, Thorax, Neck, Head, LShoulder, LElbow, LWrist,
//             RShoulder, RElbow, RWrist

function mediapipeToH36M(landmarks) {
  const lm = landmarks
  const avg = (a, b) => ({
    x: (lm[a].x + lm[b].x) / 2,
    y: (lm[a].y + lm[b].y) / 2,
    z: (lm[a].z + lm[b].z) / 2,
  })

  const hipMid      = avg(23, 24)
  const shoulderMid = avg(11, 12)
  const earMid      = avg(7, 8)

  const neck = {
    x: (shoulderMid.x + earMid.x) / 2,
    y: (shoulderMid.y + earMid.y) / 2,
    z: (shoulderMid.z + earMid.z) / 2,
  }
  const spine = {
    x: (hipMid.x + shoulderMid.x) / 2,
    y: (hipMid.y + shoulderMid.y) / 2,
    z: (hipMid.z + shoulderMid.z) / 2,
  }

  return [
    hipMid,       // 0  Hip (root)
    lm[24],       // 1  RHip
    lm[26],       // 2  RKnee
    lm[28],       // 3  RAnkle
    lm[23],       // 4  LHip
    lm[25],       // 5  LKnee
    lm[27],       // 6  LAnkle
    spine,        // 7  Spine
    shoulderMid,  // 8  Thorax
    neck,         // 9  Neck
    earMid,       // 10 Head
    lm[11],       // 11 LShoulder
    lm[13],       // 12 LElbow
    lm[15],       // 13 LWrist
    lm[12],       // 14 RShoulder
    lm[14],       // 15 RElbow
    lm[16],       // 16 RWrist
  ]
}

// ── H36M 17 back to MediaPipe 33 ─────────────────────────────────────────────
function h36mToMediapipe(original33, h36m17, scaleFactor) {
  const out = original33.map(lm => ({ ...lm }))

  const pairs = [
    [23, 4],  // LHip
    [24, 1],  // RHip
    [25, 5],  // LKnee
    [26, 2],  // RKnee
    [27, 6],  // LAnkle
    [28, 3],  // RAnkle
    [11, 11], // LShoulder
    [12, 14], // RShoulder
    [13, 12], // LElbow
    [14, 15], // RElbow
    [15, 13], // LWrist
    [16, 16], // RWrist
  ]

  for (const [mpIdx, h36Idx] of pairs) {
    out[mpIdx] = {
      ...out[mpIdx],
      z: h36m17[h36Idx].z * scaleFactor,
    }
  }

  return out
}

// ── Normalise H36M joints for model input ────────────────────────────────────
function normaliseH36M(joints17) {
  const root = joints17[0]
  const normalised = joints17.map(j => ({
    x: j.x - root.x,
    y: j.y - root.y,
  }))

  const dists = normalised.slice(1).map(j => Math.sqrt(j.x ** 2 + j.y ** 2))
  const meanDist = dists.reduce((a, b) => a + b, 0) / dists.length || 1
  const scale = 1 / meanDist

  return normalised.map(j => ({ x: j.x * scale, y: j.y * scale }))
}

// ── Session singleton ─────────────────────────────────────────────────────────
let session = null

async function getSession(onProgress) {
  if (session) return session

  onProgress?.('Loading 3D model…')

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

  const modelPath = `${import.meta.env.BASE_URL}models/mhformer_NxFxKxXY_1x27x17x2.onnx`
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  onProgress?.('Model ready')
  return session
}

// ── Main export ───────────────────────────────────────────────────────────────
export async function liftPosesTo3D(frames, onProgress) {
  const sess     = await getSession(onProgress)
  const F        = 27
  const N        = frames.length
  const improved = frames.map(f => ({ ...f }))

  onProgress?.(`Lifting ${N} frames to 3D…`)

  let scaleFactor  = 1.0
  let scaleComputed = false

  const mp0 = frames[0].worldLandmarks ?? frames[0].landmarks
  const mpHipWidth = Math.abs(mp0[24].x - mp0[23].x)

  for (let i = 0; i < N; i++) {
    const window = []
    for (let w = 0; w < F; w++) {
      const idx  = Math.min(Math.max(i - Math.floor(F / 2) + w, 0), N - 1)
      const src  = frames[idx].landmarks
      const h36m = mediapipeToH36M(src)
      const norm = normaliseH36M(h36m)
      window.push(norm)
    }

    const inputData = new Float32Array(1 * F * 17 * 2)
    for (let w = 0; w < F; w++) {
      for (let j = 0; j < 17; j++) {
        inputData[(w * 17 + j) * 2 + 0] = window[w][j].x
        inputData[(w * 17 + j) * 2 + 1] = window[w][j].y
      }
    }

    const tensor  = new ort.Tensor('float32', inputData, [1, F, 17, 2])
    const feeds   = { [sess.inputNames[0]]: tensor }
    const results = await sess.run(feeds)

    const outData  = results[sess.outputNames[0]].data
    const lifted17 = []
    for (let j = 0; j < 17; j++) {
      lifted17.push({
        x: outData[j * 3 + 0],
        y: outData[j * 3 + 1],
        z: outData[j * 3 + 2],
      })
    }

    if (!scaleComputed) {
      const mhHipWidth = Math.abs(lifted17[1].x - lifted17[4].x)
      scaleFactor  = mhHipWidth > 0.001 ? mpHipWidth / mhHipWidth : 1.0
      scaleComputed = true
    }

    const original33 = frames[i].worldLandmarks ?? frames[i].landmarks
    improved[i] = {
      ...frames[i],
      worldLandmarks: h36mToMediapipe(original33, lifted17, scaleFactor),
    }

    if (i % 10 === 0) onProgress?.(`Lifting poses… ${Math.round((i / N) * 100)}%`)
  }

  onProgress?.('Done')
  return improved
}