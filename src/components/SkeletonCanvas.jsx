import { useEffect, useRef } from 'react'
import { POSE_CONNECTIONS, NAMED_JOINTS, PERSON_COLORS } from '../hooks/usePoseExtractor'

// When rendering multiple people, each gets a fixed colour from PERSON_COLORS.
// When rendering a single person, jointColor() picks by body region.
function jointColor(index) {
  if ([11,12,13,14,15,16,17,18,19,20,21,22].includes(index)) return '#7c6cff'
  if ([23,24,25,26,27,28,29,30,31,32].includes(index)) return '#39e8a0'
  return '#f5a623'
}

function drawSkeleton(ctx, landmarks, width, height, color = null) {
  function lx(lm) { return lm.x * width }
  function ly(lm) { return lm.y * height }

  // Bones
  POSE_CONNECTIONS.forEach(([i, j]) => {
    const a = landmarks[i]
    const b = landmarks[j]
    if (!a || !b) return
    const minVis = Math.min(a.v ?? 1, b.v ?? 1)
    if (minVis < 0.3) return
    ctx.beginPath()
    ctx.moveTo(lx(a), ly(a))
    ctx.lineTo(lx(b), ly(b))
    ctx.strokeStyle = color
      ? `${color}${Math.round((0.4 + minVis * 0.5) * 255).toString(16).padStart(2,'0')}`
      : `rgba(80,80,120,${0.4 + minVis * 0.5})`
    ctx.lineWidth = 1.5
    ctx.stroke()
  })

  // Joints
  landmarks.forEach((lm, i) => {
    if ((lm.v ?? 1) < 0.3) return
    const x = lx(lm)
    const y = ly(lm)
    const isNamed = NAMED_JOINTS[i] !== undefined
    const r = isNamed ? 4 : 2.5
    const c = color ?? jointColor(i)

    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fillStyle = c
    ctx.shadowColor = c
    ctx.shadowBlur = isNamed ? 6 : 0
    ctx.fill()
    ctx.shadowBlur = 0

    if (isNamed && !color) {
      ctx.fillStyle = 'rgba(136,136,170,0.8)'
      ctx.font = '9px DM Mono, monospace'
      ctx.fillText(NAMED_JOINTS[i], x + 6, y - 4)
    }
  })
}

// Returns the person index whose hip center is closest to the click point,
// or -1 if nothing is close enough
function hitTestPersons(persons, clickX, clickY, width, height) {
  const HIT_RADIUS = 40 // px
  let bestDist = HIT_RADIUS
  let bestIdx  = -1

  persons.forEach((landmarks, idx) => {
    const l = landmarks[23]
    const r = landmarks[24]
    if (!l || !r) return
    const cx = ((l.x + r.x) / 2) * width
    const cy = ((l.y + r.y) / 2) * height
    const d  = Math.sqrt((clickX - cx) ** 2 + (clickY - cy) ** 2)
    if (d < bestDist) { bestDist = d; bestIdx = idx }

    // Also test all named joints as click targets
    Object.keys(NAMED_JOINTS).forEach((ji) => {
      const lm = landmarks[parseInt(ji)]
      if (!lm) return
      const jx = lm.x * width
      const jy = lm.y * height
      const jd = Math.sqrt((clickX - jx) ** 2 + (clickY - jy) ** 2)
      if (jd < bestDist) { bestDist = jd; bestIdx = idx }
    })
  })

  return bestIdx
}


// Main Component
export default function SkeletonCanvas({
  // Single-person mode (frame inspector)
  landmarks,
  // Multi-person selection mode
  persons,
  onSelectPerson,
  // Shared
  width = 400,
  height = 400,
}) {
  const canvasRef = useRef(null)
  const isMulti   = Array.isArray(persons) && persons.length > 0

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const dpr = window.devicePixelRatio || 1
    canvas.width  = width  * dpr
    canvas.height = height * dpr
    canvas.style.width  = width  + 'px'
    canvas.style.height = height + 'px'
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, width, height)

    // Grid
    ctx.strokeStyle = 'rgba(42,42,58,0.6)'
    ctx.lineWidth = 0.5
    for (let x = 0; x <= width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke()
    }
    for (let y = 0; y <= height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke()
    }

    if (isMulti) {
      // Draw all detected persons with their assigned colours
      persons.forEach((lms, idx) => {
        drawSkeleton(ctx, lms, width, height, PERSON_COLORS[idx % PERSON_COLORS.length])
      })

      // Legend and click instruction
      persons.forEach((_, idx) => {
        const c = PERSON_COLORS[idx % PERSON_COLORS.length]
        ctx.fillStyle = c
        ctx.fillRect(8, 8 + idx * 18, 10, 10)
        ctx.fillStyle = 'rgba(136,136,170,0.9)'
        ctx.font = '10px DM Mono, monospace'
        ctx.fillText(`Person ${idx + 1}`, 24, 17 + idx * 18)
      })

      ctx.fillStyle = 'rgba(74,74,106,0.8)'
      ctx.font = '10px DM Mono, monospace'
      ctx.fillText('click a skeleton to track', 8, height - 8)

    } else if (landmarks) {
      // Single-person mode
      drawSkeleton(ctx, landmarks, width, height, null)
      ctx.fillStyle = 'rgba(74,74,106,0.7)'
      ctx.font = '10px DM Mono, monospace'
      ctx.fillText(`${landmarks.length} pts`, 8, height - 8)
    }

  }, [landmarks, persons, width, height])

  function handleClick(e) {
    if (!isMulti || !onSelectPerson) return
    const canvas = canvasRef.current
    const rect   = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top
    const idx    = hitTestPersons(persons, clickX, clickY, width, height)
    if (idx !== -1) onSelectPerson(idx)
  }

  if (!isMulti && !landmarks) {
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
      onClick={handleClick}
      style={{
        display: 'block',
        borderRadius: 4,
        border: '1px solid #2a2a3a',
        cursor: isMulti ? 'crosshair' : 'default',
      }}
    />
  )
}