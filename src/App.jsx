import { useRef, useState } from 'react'
import { usePoseExtractor, DEFAULT_SETTINGS } from './hooks/usePoseExtractor'
import FrameInspector from './components/FrameInspector'
import PersonSelector from './components/PersonSelector'
import PoseEditor3D from './components/PoseEditor3D'
import { exportBVH } from './utils/exportBVH'
import './App.css'

const STATUS_COLOR = {
  idle:            '#5b5b78',
  'loading-model': '#f5a623',
  prescanning:     '#f5a623',
  'select-person': '#f5a623',
  processing:      '#7c6cff',
  done:            '#39e8a0',
  error:           '#ff5d73',
}

const STATUS_LABEL = {
  idle:            '● Idle',
  'loading-model': '◌ Loading model…',
  prescanning:     '◌ Pre-scanning for people…',
  'select-person': '◎ Select a person to track',
  processing:      '◎ Processing…',
  done:            '✓ Done',
  error:           '✕ Error',
}

function StatBox({ label, value, unit }) {
  return (
    <div className="stat-box">
      <div className="stat-box__label">{label}</div>
      <div className="stat-box__value">
        {value}
        {unit && <span className="stat-box__unit">{unit}</span>}
      </div>
    </div>
  )
}

function Switch({ checked, disabled, onChange }) {
  return (
    <label className="switch">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} />
      <span className="switch__slider" />
    </label>
  )
}

export default function App() {
  const {
    preScan, processImage, processVideo, cancelProcessing, clearResults,
    applyFrameEdit, resetFrame,
    status, progress,
    frames, scanFrames,
    stats, error,
    fileRef, selectPersonRef, statsSummaryRef
  } = usePoseExtractor()

  const [dragOver, setDragOver] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [lastFile, setLastFile] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const [editingFrame, setEditingFrame] = useState(null)

  const pendingFileRef = useRef(null)
  const inputRef = useRef(null)

  function setSetting(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function handleFile(file) {
    if (!file) return

    // Image: skip the modal, go straight to single-frame extraction.
    if (file.type.startsWith('image/')) {
      setLastFile(file)
      fileRef.current = file
      processImage(file, settings)
      return
    }

    if (file.type.startsWith('video/')) {
      pendingFileRef.current = file
      setShowModal(true)
    }
  }

  function handleSinglePerson() {
    const file = pendingFileRef.current
    setShowModal(false)
    setLastFile(file)
    fileRef.current = file
    processVideo(null, settings)
  }

  function handleMultiPerson() {
    const file = pendingFileRef.current
    setShowModal(false)
    setLastFile(file)
    preScan(file)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  // Called by PersonSelector when the user clicks a skeleton.
  function handlePersonSelected(seed) {
    processVideo(seed, settings)
  }

  function exportJSON() {
    const blob = new Blob([JSON.stringify(frames, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pose_landmarks.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const isProcessing = status === 'processing' || status === 'loading-model' || status === 'prescanning'
  const isSelecting  = status === 'select-person'

  const uploadZoneClass = [
    'upload-zone',
    dragOver     ? 'upload-zone--drag' : '',
    isProcessing ? 'upload-zone--busy' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className="app">

      {/* Header */}
      <div className="header">
        <div className="header__title-row">
          <span className="header__mark">⛷</span>
          <h1 className="header__title">PoseFinder</h1>
        </div>
        <p className="header__desc">
          Upload a video or image and turn it into ready-to-use pose animation data.
        </p>
      </div>

      {/* Status bar */}
      <div className="status-bar">
        <span className="status-bar__label" style={{ color: STATUS_COLOR[status] }}>
          {STATUS_LABEL[status]}
        </span>
        {isProcessing && (
          <>
            <div className="status-bar__track">
              <div className="status-bar__fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="status-bar__pct">{progress}%</span>
          </>
        )}
      </div>

      {/* Settings panel */}
      <div className="panel">
        <div className="panel__head">
          <span className="panel__title">Extraction settings</span>
          <button
            className="btn btn--ghost"
            onClick={() => setSettings(DEFAULT_SETTINGS)}
            disabled={isProcessing}
          >
            Reset defaults
          </button>
        </div>

        {/* Primary settings — the choices that most change the output */}
        <div className="option-row">
          <div className="option-row__text">
            <span className="option-row__label">Model quality</span>
            <span className="option-row__hint">
              Heavier models track harder poses more accurately, but process slower.
            </span>
          </div>
          <div className="option-row__control">
            <select
              className="select"
              disabled={isProcessing}
              value={settings.modelQuality}
              onChange={(e) => setSetting('modelQuality', e.target.value)}
            >
              <option value="lite">Lite — fastest</option>
              <option value="full">Full — balanced (default)</option>
              <option value="heavy">Heavy — most accurate</option>
            </select>
          </div>
        </div>

        <div className="option-row">
          <div className="option-row__text">
            <span className="option-row__label">Track hands &amp; fingers</span>
            <span className="option-row__hint">
              Adds finger poses to the skeleton. Turn off to process faster — it skips a model pass on every frame.
            </span>
          </div>
          <div className="option-row__control">
            <Switch
              checked={settings.trackHands}
              disabled={isProcessing}
              onChange={(e) => setSetting('trackHands', e.target.checked)}
            />
          </div>
        </div>

        {/* Advanced settings — sensible defaults, tucked away */}
        <details className="advanced">
          <summary className="advanced__summary">
            <span className="advanced__chevron">▶</span>
            Advanced settings
          </summary>

          <div className="settings__grid">
            <div className="setting">
              <div className="setting__header">
                <span className="setting__label">Capture fps</span>
                <span className="setting__value">{settings.captureFps} fps</span>
              </div>
              <input type="range" min={1} max={30} step={1}
                disabled={isProcessing}
                value={settings.captureFps}
                onChange={(e) => setSetting('captureFps', Number(e.target.value))}
              />
              <span className="setting__hint">
                How often to sample the video — lower = fewer duplicate poses in lower fps videos
              </span>
            </div>

            <div className="setting">
              <div className="setting__header">
                <span className="setting__label">Confidence threshold</span>
                <span className="setting__value">{Math.round(settings.confidenceThreshold * 100)}%</span>
              </div>
              <input type="range" min={0.1} max={0.95} step={0.05}
                disabled={isProcessing}
                value={settings.confidenceThreshold}
                onChange={(e) => setSetting('confidenceThreshold', Number(e.target.value))}
              />
              <span className="setting__hint">
                Drop frames where the positions of joints are uncertain — higher = fewer, more accurate frames
              </span>
            </div>

            <div className="setting">
              <div className="setting__header">
                <span className="setting__label">Keyframe sensitivity</span>
                <span className="setting__value">{settings.keyframeThreshold.toFixed(2)}</span>
              </div>
              <input type="range" min={0.01} max={0.2} step={0.01}
                disabled={isProcessing}
                value={settings.keyframeThreshold}
                onChange={(e) => setSetting('keyframeThreshold', Number(e.target.value))}
              />
              <span className="setting__hint">
                Minimum pose difference to count as a new keyframe — higher = fewer, more unique poses
              </span>
            </div>

            <div className="setting">
              <div className="setting__header">
                <span className="setting__label">Max frames</span>
                <span className="setting__value">{settings.maxFrames}</span>
              </div>
              <input type="range" min={10} max={1000} step={5}
                disabled={isProcessing}
                value={settings.maxFrames}
                onChange={(e) => setSetting('maxFrames', Number(e.target.value))}
              />
              <span className="setting__hint">
                Hard frame cap — spreads frames evenly if total number of keyframes exceeds this cap
              </span>
            </div>
          </div>

          {/* Export-time toggles — applied at export, no reprocessing needed */}
          <div className="option-row">
            <div className="option-row__text">
              <span className="option-row__label">Keep feet planted</span>
              <span className="option-row__hint">
                Locks the feet to a stable floor so they don't float or sink. Turn off for aerial / high-action clips.
              </span>
            </div>
            <div className="option-row__control">
              <Switch
                checked={settings.keepFeetPlanted}
                onChange={(e) => setSetting('keepFeetPlanted', e.target.checked)}
              />
            </div>
          </div>

          <div className="option-row">
            <div className="option-row__text">
              <span className="option-row__label">Strict anatomical limits</span>
              <span className="option-row__hint">
                Tightens joint limits (feet, wrists) so limbs stay more rigid and neutral, at the cost of some fidelity.
              </span>
            </div>
            <div className="option-row__control">
              <Switch
                checked={settings.strictAnatomy}
                onChange={(e) => setSetting('strictAnatomy', e.target.checked)}
              />
            </div>
          </div>

          <div className="option-row">
            <div className="option-row__text">
              <span className="option-row__label">Preserve turning (body facing)</span>
              <span className="option-row__hint">
                Lets the root rotate so the character turns to face the way the subject does. Turn off to stabilise facing (root yaw locked).
              </span>
            </div>
            <div className="option-row__control">
              <Switch
                checked={settings.preserveFacing}
                onChange={(e) => setSetting('preserveFacing', e.target.checked)}
              />
            </div>
          </div>
        </details>
      </div>

      {/* Upload zone */}
      <div
        className={uploadZoneClass}
        onClick={() => !isProcessing && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,image/jpeg,image/png,image/webp"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="upload-zone__icon">⬆</div>
        <div className="upload-zone__label">
          {isProcessing ? 'Processing…' : 'Drop a video or click to upload'}
        </div>
        <div className="upload-zone__hint">MP4, MOV, WebM · JPG, PNG, WebP</div>
      </div>

      {/* Action row */}
      <div className="actions">
        {isProcessing && (
          <button className="btn btn--danger" onClick={cancelProcessing}>
            ✕ Cancel
          </button>
        )}
        {lastFile && !isProcessing && (
          <button
            className="btn btn--accent"
            onClick={() => { pendingFileRef.current = lastFile; setShowModal(true) }}
          >
            ↺ Rescan
          </button>
        )}
      </div>

      {/* Multi-person modal, shown right after a video is dropped */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal__title">Are there several people in this video?</div>
            <p className="modal__desc">
              Single-person videos process faster and more accurately.<br />
              Choose multi-person to manually select who to track.
            </p>
            <div className="modal__actions">
              <button className="btn btn--green" onClick={handleSinglePerson}>
                Just one person
              </button>
              <button className="btn btn--accent" onClick={handleMultiPerson}>
                Multiple people
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && <div className="error-banner">{error}</div>}

      {/* Person selector filmstrip */}
      <div ref={selectPersonRef}>
        {isSelecting && scanFrames.length > 0 && (
          <div id="select-person-section">
            <PersonSelector
              scanFrames={scanFrames}
              onSelect={handlePersonSelected}
            />
          </div>
        )}
      </div>

      {/* Stats */}
      <div ref={statsSummaryRef}>
        {stats && (
          <div id="stats-summary-section" className="stats-row">
            <StatBox label="Sampled"   value={stats.totalSampled}   unit={`@ ${stats.captureFps}fps`} />
            <StatBox label="Confident" value={stats.capturedCount} unit="frames" />
            <StatBox label="Keyframes" value={stats.keyframeCount} unit="unique" />
            <StatBox label="Final"     value={stats.frameCount}    unit="kept" />
          </div>
        )}
      </div>

      {/* Frame inspector */}
      {frames.length > 0 && (
        <>
          <div className="inspector-header">
            <h2 className="inspector-header__title">Frame Inspector</h2>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn btn--amber"
                onClick={() => exportBVH(frames, {
                  captureFps:      stats.captureFps,
                  boneLengths:     stats.boneLengths,
                  modelQuality:    stats.modelQuality,
                  keepFeetPlanted: settings.keepFeetPlanted,
                  strictAnatomy:   settings.strictAnatomy,
                  preserveFacing:  settings.preserveFacing,
                })}
              >
                ↓ Export BVH
              </button>
              <button className="btn btn--green" onClick={exportJSON}>
                ↓ Export JSON
              </button>
              <button
                className="btn btn--ghost"
                onClick={() => { setLastFile(null); clearResults() }}
                title="Remove the saved result"
              >
                ✕ Clear
              </button>
            </div>
          </div>
          <div className="inspector-card">
            <FrameInspector frames={frames} stats={stats} onEdit={setEditingFrame} />
          </div>
        </>
      )}

      {/* Full-screen 3D pose editor */}
      {editingFrame !== null && frames.length > 0 && (
        <PoseEditor3D
          frames={frames}
          startIndex={editingFrame}
          onApplyEdit={applyFrameEdit}
          onResetFrame={resetFrame}
          onClose={() => setEditingFrame(null)}
        />
      )}

    </div>
  )
}
