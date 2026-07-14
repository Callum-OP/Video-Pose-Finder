import { useEffect, useRef, useState, useCallback } from 'react'
import { PoseEditorScene } from './PoseEditorScene'
import { buildControlPositions, writeBackFrame, EDIT_TARGET_BY_KEY } from '../utils/poseEditMath'

// How many recent pose changes the editor's undo history keeps.
const MAX_UNDO = 30

// ── Full-screen 3D pose editor ────────────────────────────────────────────────
// Shows the captured pose on a rigged humanoid (or a capsule mannequin fallback),
// lets the user rotate/move joints per frame, and commits edits back into `frames`
// via onApplyEdit so Export BVH/JSON pick them up unchanged.

const s = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 1000,
    background: '#07070b', display: 'flex', flexDirection: 'column',
    fontFamily: 'DM Mono, monospace', color: '#eeeef5',
  },
  topbar: {
    display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
    borderBottom: '1px solid #1f1f2e', flexWrap: 'wrap',
  },
  title: { fontSize: 14, fontWeight: 600, marginRight: 8 },
  spacer: { flex: 1 },
  viewport: { flex: 1, position: 'relative', minHeight: 0 },
  badge: {
    position: 'absolute', top: 12, left: 12, padding: '6px 10px',
    background: 'rgba(20,20,32,0.8)', border: '1px solid #2a2a3a', borderRadius: 6,
    fontSize: 12, pointerEvents: 'none',
  },
  hint: {
    position: 'absolute', bottom: 12, left: 12, right: 12, fontSize: 11,
    color: '#8888aa', pointerEvents: 'none', lineHeight: 1.5,
  },
  bottombar: {
    display: 'flex', alignItems: 'center', gap: 14, padding: '10px 16px',
    borderTop: '1px solid #1f1f2e',
  },
  scrubber: { flex: 1, accentColor: '#7c6cff', cursor: 'pointer' },
  frameInfo: { fontSize: 12, color: '#8888aa', whiteSpace: 'nowrap', minWidth: 110, textAlign: 'center' },
  seg: { display: 'inline-flex', border: '1px solid #2a2a3a', borderRadius: 6, overflow: 'hidden' },
  segBtn: (active) => ({
    padding: '6px 12px', fontSize: 12, cursor: 'pointer', border: 'none',
    background: active ? '#7c6cff' : 'transparent', color: active ? '#0a0a0f' : '#aaaac8',
    fontFamily: 'inherit', fontWeight: active ? 600 : 400,
  }),
  toggle: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#aaaac8', cursor: 'pointer' },
}

export default function PoseEditor3D({ frames, startIndex = 0, onClose, onApplyEdit, onResetFrame }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const [frameIdx, setFrameIdx] = useState(Math.min(startIndex, frames.length - 1))
  const [tool, setTool] = useState('rotate')
  const [showMesh, setShowMesh] = useState(true)
  const [selected, setSelected] = useState(null)
  const [playing, setPlaying] = useState(false)

  // Latest values for the stable scene callbacks.
  const frameIdxRef = useRef(frameIdx)
  const framesRef = useRef(frames)
  frameIdxRef.current = frameIdx
  framesRef.current = frames

  // ── Undo / redo history (per editor session) ──────────────────────────────
  // Each committed change (a drag, a rotate, a reset) pushes the frame's previous
  // pose so it can be restored. Capped at MAX_UNDO recent changes.
  const historyRef = useRef({ past: [], future: [] })
  const [hist, setHist] = useState({ past: 0, future: 0 })

  // Snapshot a frame's current pose (immutable arrays — cheap to keep).
  const snap = useCallback((i) => {
    const f = framesRef.current[i]
    return f ? { frameIndex: i, landmarks: f.landmarks, worldLandmarks: f.worldLandmarks } : null
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Record the pre-change state of frame i (call right before applying a change).
  const recordHistory = useCallback((i) => {
    const s0 = snap(i)
    if (!s0) return
    const h = historyRef.current
    h.past.push(s0)
    if (h.past.length > MAX_UNDO) h.past.shift()
    h.future = []
    setHist({ past: h.past.length, future: 0 })
  }, [snap])

  const undo = useCallback(() => {
    const h = historyRef.current
    if (!h.past.length) return
    const entry = h.past.pop()
    const cur = snap(entry.frameIndex)
    if (cur) h.future.push(cur)
    onApplyEdit(entry.frameIndex, { landmarks: entry.landmarks, worldLandmarks: entry.worldLandmarks })
    setFrameIdx(entry.frameIndex)
    setHist({ past: h.past.length, future: h.future.length })
  }, [snap, onApplyEdit])

  const redo = useCallback(() => {
    const h = historyRef.current
    if (!h.future.length) return
    const entry = h.future.pop()
    const cur = snap(entry.frameIndex)
    if (cur) { h.past.push(cur); if (h.past.length > MAX_UNDO) h.past.shift() }
    onApplyEdit(entry.frameIndex, { landmarks: entry.landmarks, worldLandmarks: entry.worldLandmarks })
    setFrameIdx(entry.frameIndex)
    setHist({ past: h.past.length, future: h.future.length })
  }, [snap, onApplyEdit])

  // Create the scene once.
  useEffect(() => {
    const scene = new PoseEditorScene(containerRef.current, {
      onEdit: (pos) => {
        const i = frameIdxRef.current
        const frame = framesRef.current[i]
        if (!frame) return
        recordHistory(i)
        onApplyEdit(i, writeBackFrame(frame, pos))
      },
      onSelect: (key) => setSelected(key),
    })
    sceneRef.current = scene
    return () => { scene.dispose(); sceneRef.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Push the current frame's pose into the scene whenever the frame (or its data) changes.
  useEffect(() => {
    const scene = sceneRef.current
    const frame = frames[frameIdx]
    if (!scene || !frame) return
    scene.setPose(buildControlPositions(frame), frame.handData)
  }, [frameIdx, frames])

  useEffect(() => { sceneRef.current?.setTool(tool) }, [tool])
  useEffect(() => { sceneRef.current?.setShowMesh(showMesh) }, [showMesh])

  // ── Playback ───────────────────────────────────────────────────────────────
  // Steps through the sequence using each frame's captured timing (clamped so
  // dropped-frame gaps don't stall playback), looping back to the start.
  useEffect(() => {
    if (!playing) return
    if (frames.length < 2) { setPlaying(false); return }
    const cur = frames[frameIdx]
    const next = frames[(frameIdx + 1) % frames.length]
    const delta = Math.max(16, Math.min(500, (next?.timeMs ?? 0) - (cur?.timeMs ?? 0)))
    const t = setTimeout(() => setFrameIdx((i) => (i + 1) % frames.length), delta)
    return () => clearTimeout(t)
  }, [playing, frameIdx, frames])

  const togglePlay = useCallback(() => {
    if (framesRef.current.length < 2) return
    setPlaying((p) => {
      // Deselect when starting playback so the gizmo doesn't fight the animation.
      if (!p) { setSelected(null); sceneRef.current?.select(null) }
      return !p
    })
  }, [])

  // Keyboard: Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, Space play/pause,
  // arrows cycle frames, Esc closes.
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.ctrlKey || e.metaKey
      const k = e.key.toLowerCase()
      if (mod && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return }
      if (mod && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return }
      if (e.key === ' ') { e.preventDefault(); togglePlay() }
      else if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight') step(1)
      else if (e.key === 'ArrowLeft') step(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames.length, undo, redo, togglePlay])

  function step(d) {
    setPlaying(false)
    setFrameIdx((i) => Math.max(0, Math.min(frames.length - 1, i + d)))
  }

  const frame = frames[frameIdx]
  const selLabel = selected ? (EDIT_TARGET_BY_KEY[selected]?.label ?? selected) : null

  return (
    <div style={s.overlay}>
      {/* Top toolbar */}
      <div style={s.topbar}>
        <span style={s.title}>✎ Pose Editor</span>

        <div style={s.seg}>
          <button style={s.segBtn(tool === 'rotate')} onClick={() => setTool('rotate')}>⟳ Rotate</button>
          <button style={s.segBtn(tool === 'move')} onClick={() => setTool('move')}>✥ Move</button>
        </div>

        <label style={s.toggle}>
          <input type="checkbox" checked={showMesh} onChange={(e) => setShowMesh(e.target.checked)} />
          Character mesh
        </label>

        <div style={s.spacer} />

        <button
          className="btn btn--ghost"
          onClick={undo}
          disabled={hist.past === 0}
          title="Undo (Ctrl+Z)"
        >
          ↶ Undo
        </button>
        <button
          className="btn btn--ghost"
          onClick={redo}
          disabled={hist.future === 0}
          title="Redo (Ctrl+Y)"
        >
          ↷ Redo
        </button>
        <button
          className="btn btn--ghost"
          onClick={() => { recordHistory(frameIdx); onResetFrame(frameIdx); setSelected(null); sceneRef.current?.select(null) }}
          title="Restore this frame to its originally-captured pose"
        >
          ↺ Reset frame
        </button>
        <button className="btn btn--accent" onClick={onClose}>✓ Done</button>
      </div>

      {/* 3D viewport */}
      <div style={s.viewport}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
        <div style={s.badge}>
          {selLabel ? <>Selected: <b>{selLabel}</b></> : 'Click a joint to select'}
        </div>
        <div style={s.hint}>
          Drag to orbit · scroll to zoom · right-drag to pan.{' '}
          {tool === 'rotate'
            ? 'Select a joint and use the gizmo rings to rotate the limb — drag inside the rings for free/diagonal rotation. Bone lengths preserved (exact on export).'
            : 'Drag any joint dot in any direction to move it. (Move may be re-constrained on export; rotate is exact.)'}
          {' '}Feet planting/grounding is applied at export.
        </div>
      </div>

      {/* Bottom frame navigation */}
      <div style={s.bottombar}>
        {frames.length > 1 && (
          <button
            className={playing ? 'btn btn--accent' : 'btn btn--ghost'}
            onClick={togglePlay}
            title="Play/pause the pose sequence (Space)"
          >
            {playing ? '❚❚ Pause' : '▶ Play'}
          </button>
        )}
        <button className="btn btn--ghost" onClick={() => step(-1)} disabled={frameIdx === 0}>◄ Prev</button>
        <input
          type="range" min={0} max={frames.length - 1} value={frameIdx}
          onChange={(e) => { setPlaying(false); setFrameIdx(Number(e.target.value)) }}
          style={s.scrubber}
        />
        <button className="btn btn--ghost" onClick={() => step(1)} disabled={frameIdx === frames.length - 1}>Next ►</button>
        <span style={s.frameInfo}>
          frame {frameIdx + 1} / {frames.length}
          {frame ? ` · t=${(frame.timeMs / 1000).toFixed(2)}s` : ''}
        </span>
      </div>
    </div>
  )
}
