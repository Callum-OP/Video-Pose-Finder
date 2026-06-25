import { useEffect, useRef, useState } from 'react'
import { PoseEditorScene } from './PoseEditorScene'
import { buildControlPositions, writeBackFrame, EDIT_TARGET_BY_KEY } from '../utils/poseEditMath'

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

  // Latest values for the stable scene callbacks.
  const frameIdxRef = useRef(frameIdx)
  const framesRef = useRef(frames)
  frameIdxRef.current = frameIdx
  framesRef.current = frames

  // Create the scene once.
  useEffect(() => {
    const scene = new PoseEditorScene(containerRef.current, {
      onEdit: (pos) => {
        const i = frameIdxRef.current
        const frame = framesRef.current[i]
        if (!frame) return
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
    scene.setPose(buildControlPositions(frame))
  }, [frameIdx, frames])

  useEffect(() => { sceneRef.current?.setTool(tool) }, [tool])
  useEffect(() => { sceneRef.current?.setShowMesh(showMesh) }, [showMesh])

  // Keyboard: arrows cycle frames, Esc closes.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight') step(1)
      else if (e.key === 'ArrowLeft') step(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames.length])

  function step(d) {
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
          onClick={() => { onResetFrame(frameIdx); setSelected(null); sceneRef.current?.select(null) }}
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
          Drag to orbit · scroll to zoom · right-drag to pan. Select a joint, then use the gizmo to{' '}
          {tool === 'rotate' ? 'rotate the limb (bone lengths preserved — exact on export).'
                             : 'move the joint (may be re-constrained on export; rotate is exact).'}
          {' '}Feet planting/grounding is applied at export.
        </div>
      </div>

      {/* Bottom frame navigation */}
      <div style={s.bottombar}>
        <button className="btn btn--ghost" onClick={() => step(-1)} disabled={frameIdx === 0}>◄ Prev</button>
        <input
          type="range" min={0} max={frames.length - 1} value={frameIdx}
          onChange={(e) => setFrameIdx(Number(e.target.value))}
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
