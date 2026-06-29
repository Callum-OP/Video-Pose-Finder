import { useState } from 'react'
import { NAMED_JOINTS } from '../hooks/usePoseExtractor'
import SkeletonCanvas from './SkeletonCanvas'

const s = {
  root: {
    display: 'flex',
    gap: 16,
    flexWrap: 'wrap',
  },
  left: {
    flex: '0 0 auto',
  },
  right: {
    flex: '1 1 260px',
    minWidth: 0,
  },
  scrubberWrap: {
    marginBottom: 12,
  },
  label: {
    color: '#8888aa',
    fontSize: 11,
    fontFamily: 'DM Mono, monospace',
    marginBottom: 4,
    display: 'block',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  },
  scrubber: {
    width: '100%',
    accentColor: '#7c6cff',
    cursor: 'pointer',
  },
  frameInfo: {
    fontFamily: 'DM Mono, monospace',
    fontSize: 12,
    color: '#8888aa',
    marginTop: 4,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontFamily: 'DM Mono, monospace',
    fontSize: 11,
  },
  th: {
    textAlign: 'left',
    color: '#4a4a6a',
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    padding: '4px 8px',
    borderBottom: '1px solid #2a2a3a',
    fontWeight: 400,
  },
  td: {
    padding: '3px 8px',
    borderBottom: '1px solid rgba(42,42,58,0.5)',
    color: '#8888aa',
    whiteSpace: 'nowrap',
  },
  tdName: {
    padding: '3px 8px',
    borderBottom: '1px solid rgba(42,42,58,0.5)',
    color: '#eeeef5',
    fontWeight: 500,
  },
  vis: (v) => ({
    display: 'inline-block',
    width: 32,
    height: 4,
    borderRadius: 2,
    background: v > 0.7 ? '#39e8a0' : v > 0.4 ? '#f5a623' : '#ff4d6d',
    opacity: 0.3 + v * 0.7,
  }),
}

export default function FrameInspector({ frames, stats, onEdit }) {
  const [frameIdx, setFrameIdx] = useState(0)

  if (!frames.length) return null

  const frame = frames[Math.min(frameIdx, frames.length - 1)]
  const namedLandmarks = Object.entries(NAMED_JOINTS).map(([i, name]) => ({
    i: parseInt(i),
    name,
    lm: frame.landmarks[parseInt(i)],
  }))

  return (
    <div style={s.root}>
      <div style={s.left}>
        <span style={s.label}>skeleton preview</span>
        <SkeletonCanvas landmarks={frame.landmarks} width={320} height={320} />
        {onEdit && (
          <button
            className="btn btn--accent"
            style={{ marginTop: 10, width: '100%' }}
            onClick={() => onEdit(frameIdx)}
          >
            ✎ Edit in 3D
          </button>
        )}
      </div>

      <div style={s.right}>
        <div style={s.scrubberWrap}>
          <span style={s.label}>frame scrubber</span>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={frameIdx}
            onChange={(e) => setFrameIdx(Number(e.target.value))}
            style={s.scrubber}
          />
          <div style={s.frameInfo}>
            frame {frameIdx + 1} / {frames.length} &nbsp;·&nbsp;
            t = {(frame.timeMs / 1000).toFixed(3)}s
          </div>
        </div>

        <span style={s.label}>key joints — frame {frameIdx + 1}</span>
        <table style={s.table}>
          <thead>
            <tr>
              <th style={s.th}>joint</th>
              <th style={s.th}>x</th>
              <th style={s.th}>y</th>
              <th style={s.th}>z</th>
              <th style={s.th}>vis</th>
            </tr>
          </thead>
          <tbody>
            {namedLandmarks.map(({ i, name, lm }) => (
              <tr key={i}>
                <td style={s.tdName}>{name}</td>
                <td style={s.td}>{lm.x.toFixed(3)}</td>
                <td style={s.td}>{lm.y.toFixed(3)}</td>
                <td style={s.td}>{lm.z.toFixed(3)}</td>
                <td style={s.td}><span style={s.vis(lm.v)} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
