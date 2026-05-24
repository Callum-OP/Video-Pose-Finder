import { useState } from 'react'
import SkeletonCanvas from './SkeletonCanvas'
import { PERSON_COLORS, hipCenter } from '../hooks/usePoseExtractor'

// Shows a filmstrip of thumbnail frames from the pre-scan.
// User clicks a skeleton on any frame to select that person.

export default function PersonSelector({ scanFrames, onSelect }) {
  const [activeFrame, setActiveFrame] = useState(0)

  if (!scanFrames.length) return null

  const frame = scanFrames[activeFrame]

  function handleSelectPerson(personIdx) {
    const landmarks = frame.persons[personIdx]
    if (!landmarks) return
    const seed = hipCenter(landmarks)
    onSelect(seed)
  }

  return (
    <div className="inspector-card" style={{ marginBottom: 24 }}>

      {/* Header */}
      <div className="inspector-header" style={{ marginBottom: 8 }}>
        <h2 className="inspector-header__title">Select person to track</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {frame.persons.map((_, idx) => (
            <button
              key={idx}
              onClick={() => handleSelectPerson(idx)}
              className="btn bg-transparent rounded-[4px] px-3.5 py-1.5 text-xs font-mono cursor-pointer transition-colors duration-100"
              style={{
                border: `1px solid ${PERSON_COLORS[idx % PERSON_COLORS.length]}`,
                color:  PERSON_COLORS[idx % PERSON_COLORS.length],
              }}
            >
              Person {idx + 1}
            </button>
          ))}
        </div>
      </div>

      <p className="setting__hint" style={{ marginBottom: 16 }}>
        Scrub the filmstrip to find the person you want, then click their skeleton or use the buttons above.
        Showing frame {activeFrame + 1} of {scanFrames.length} · {frame.persons.length} person{frame.persons.length !== 1 ? 's' : ''} detected.
      </p>

      {/* Skeleton canvas for the active frame */}
      <SkeletonCanvas
        persons={frame.persons}
        onSelectPerson={handleSelectPerson}
        width={560}
        height={380}
      />

      {/* Filmstrip */}
      <div style={{
        display: 'flex',
        gap: 6,
        overflowX: 'auto',
        marginTop: 12,
        paddingBottom: 4,
      }}>
        {scanFrames.map((sf, idx) => (
          <div
            key={idx}
            onClick={() => setActiveFrame(idx)}
            style={{
              flex: '0 0 auto',
              cursor: 'pointer',
              border: idx === activeFrame
                ? '2px solid #7c6cff'
                : '2px solid #2a2a3a',
              borderRadius: 4,
              overflow: 'hidden',
              position: 'relative',
              opacity: idx === activeFrame ? 1 : 0.6,
              transition: 'all 0.1s ease',
            }}
          >
            <img
              src={sf.thumbnail}
              width={80}
              height={45}
              style={{ display: 'block' }}
              alt={`frame ${idx + 1}`}
            />
            {/* Person count badge */}
            {sf.persons.length > 1 && (
              <div style={{
                position: 'absolute',
                top: 2, right: 2,
                background: '#f5a623',
                color: '#0a0a0f',
                fontSize: 9,
                fontFamily: 'DM Mono, monospace',
                fontWeight: 700,
                borderRadius: 2,
                padding: '0 3px',
                lineHeight: '14px',
              }}>
                {sf.persons.length}
              </div>
            )}
            {/* Timestamp */}
            <div style={{
              position: 'absolute',
              bottom: 2, left: 2,
              color: 'rgba(238,238,245,0.7)',
              fontSize: 8,
              fontFamily: 'DM Mono, monospace',
              textShadow: '0 0 4px #000',
            }}>
              {(sf.timeMs / 1000).toFixed(1)}s
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}