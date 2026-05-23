import { useEffect, useRef } from 'react'
import { POSE_CONNECTIONS, NAMED_JOINTS } from '../hooks/usePoseExtractor'

// Joint colors by body region
function jointColor(index) {
  if ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22].includes(index)) return '#7c6cff' // arms: accent
  if ([23, 24, 25, 26, 27, 28, 29, 30, 31, 32].includes(index)) return '#39e8a0'          // legs: green
  return '#f5a623'                                                                           // face/torso: amber
}

export default function SkeletonCanvas({ landmarks, width = 400, height = 400 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !landmarks) return
    const ctx = canvas.getContext('2d')

    // Hi-DPI
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = width + 'px'
    canvas.style.height = height + 'px'
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, width, height)

    // Background
    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, width, height)

    // Grid lines
    ctx.strokeStyle = 'rgba(42,42,58,0.6)'
    ctx.lineWidth = 0.5
    for (let x = 0; x <= width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke()
    }
    for (let y = 0; y <= height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke()
    }

    // Helper: landmark coords → canvas px
    // MediaPipe normalises x/y 0→1 relative to image width/height
    function lx(lm) { return lm.x * width }
    function ly(lm) { return lm.y * height }

    // Draw bones (connections)
    POSE_CONNECTIONS.forEach(([i, j]) => {
      const a = landmarks[i]
      const b = landmarks[j]
      if (!a || !b) return

      const minVis = Math.min(a.v ?? 1, b.v ?? 1)
      if (minVis < 0.3) return // skip very uncertain joints

      ctx.beginPath()
      ctx.moveTo(lx(a), ly(a))
      ctx.lineTo(lx(b), ly(b))
      ctx.strokeStyle = `rgba(80,80,120,${0.4 + minVis * 0.5})`
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Draw joints
    landmarks.forEach((lm, i) => {
      if ((lm.v ?? 1) < 0.3) return

      const x = lx(lm)
      const y = ly(lm)
      const isNamed = NAMED_JOINTS[i] !== undefined
      const r = isNamed ? 4 : 2.5

      // Glow
      if (isNamed) {
        ctx.beginPath()
        ctx.arc(x, y, r + 4, 0, Math.PI * 2)
        const color = jointColor(i)
        ctx.fillStyle = color.replace(')', ', 0.15)').replace('rgb', 'rgba').replace('#', 'rgba(').replace(')', '')
        // Simple glow: draw larger transparent circle
        ctx.shadowColor = jointColor(i)
        ctx.shadowBlur = 8
      }

      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fillStyle = jointColor(i)
      ctx.shadowBlur = isNamed ? 6 : 0
      ctx.shadowColor = jointColor(i)
      ctx.fill()
      ctx.shadowBlur = 0

      // Label for key joints
      if (isNamed) {
        ctx.fillStyle = 'rgba(136,136,170,0.8)'
        ctx.font = '9px DM Mono, monospace'
        ctx.fillText(NAMED_JOINTS[i], x + 6, y - 4)
      }
    })

    // Corner label
    ctx.fillStyle = 'rgba(74,74,106,0.7)'
    ctx.font = '10px DM Mono, monospace'
    ctx.fillText(`${landmarks.length} pts`, 8, height - 8)

  }, [landmarks, width, height])

  if (!landmarks) {
    return (
      <div style={{
        width, height,
        background: '#0a0a0f',
        border: '1px solid #2a2a3a',
        borderRadius: 4,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#4a4a6a',
        fontFamily: 'DM Mono, monospace',
        fontSize: 12,
      }}>
        no frame selected
      </div>
    )
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', borderRadius: 4, border: '1px solid #2a2a3a' }}
    />
  )
}
