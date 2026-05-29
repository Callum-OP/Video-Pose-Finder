import { useRef, useState, useEffect } from 'react'
import { usePoseExtractor, DEFAULT_SETTINGS } from './hooks/usePoseExtractor'
import FrameInspector from './components/FrameInspector'
import PersonSelector from './components/PersonSelector'
import { exportBVH } from './utils/exportBVH'
import './App.css'

const STATUS_COLOR = {
  idle:            '#4a4a6a',
  'loading-model': '#f5a623',
  prescanning:     '#f5a623',
  'select-person': '#f5a623',
  processing:      '#6366f1',
  done:            '#22c55e',
  error:           '#ef4444',
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

export default function App() {
  // Set up backend URL
  const [enhancedMode, setEnhancedMode] = useState(false)
  const [backendAvailable, setBackendAvailable] = useState(null)

  const {
    preScan, processVideo, cancelProcessing,
    status, progress,
    frames, scanFrames,
    stats, error,
    fileRef, selectPersonRef, statsSummaryRef
  } = usePoseExtractor()

  const [dragOver, setDragOver] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [lastFile, setLastFile] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const pendingFileRef = useRef(null)
  const inputRef = useRef(null)

  function setSetting(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) return
    pendingFileRef.current = file
    setShowModal(true)
    if (inputRef.current) inputRef.current.value = ''
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

  // Called by PersonSelector when user clicks a skeleton
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

  async function exportEnhancedBVH() {
    try {
      const res = await fetch('http://localhost:8000/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
      })
      if (!res.ok) throw new Error(`Backend error: ${res.status}`)
      const data = await res.json()
      exportBVH(data.frames, stats.captureFps)
    } catch (err) {
      alert(`Enhanced export failed: ${err.message}\n\nMake sure the Python backend is running:\n  cd backend && python main.py`)
    }
  }

  const isProcessing = status === 'processing' || status === 'loading-model' || status === 'prescanning'
  const isSelecting  = status === 'select-person'

  const uploadZoneClass = [
    'upload-zone',
    dragOver     ? 'upload-zone--drag' : '',
    isProcessing ? 'upload-zone--busy' : '',
  ].filter(Boolean).join(' ')

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(r => r.ok ? setBackendAvailable(true) : setBackendAvailable(false))
      .catch(() => setBackendAvailable(false))
  }, [])

  return (
    <div className="app">

      {/* Header */}
      <div className="header">
        <div className="header__title-row">
          <h1 className="header__title">PoseFinder</h1>
        </div>
        <p className="header__desc">
          Upload a video and get pose data now.
        </p>
      </div>

      {/* Status bar */}
      <div className="status-bar">
        <span className="status-bar__label" style={{ color: STATUS_COLOR[status] }}>
          {STATUS_LABEL[status]}
        </span>
        {isProcessing && (
          <div className="status-bar__track">
            <div className="status-bar__fill" style={{ width: `${progress}%` }} />
          </div>
        )}
        {isProcessing && (
          <span className="status-bar__pct">{progress}%</span>
        )}
      </div>

      {/* Settings panel */}
      <div className="settings">
        <div className="settings__title-row">
          <span className="settings__title">Extraction settings</span>
          <button
            onClick={() => setSettings(DEFAULT_SETTINGS)}
            className="btn bg-transparent border border-rose-500 rounded-[4px] px-3.5 py-1.5 text-rose-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-rose-500/10"
          >
            Reset Defaults
          </button>
        </div>
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
      </div>

      {/* Upload row */}
      <div className="upload-row">
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
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className="upload-zone__icon">⬆</div>
          <div className="upload-zone__label">
            {isProcessing ? 'Processing…' : 'Drop a video or click to upload'}
          </div>
          <div className="upload-zone__hint">MP4, MOV, WebM</div>
        </div>
      </div>

      {/* Action Controls Row */}
      <div className="flex justify-end gap-3 mb-6 min-h-[32px]">
        {isProcessing && (
          <button
            className="btn bg-transparent border border-red-500 rounded-[4px] px-3.5 py-1.5 text-red-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-red-500/10"
            onClick={cancelProcessing}
          >
            ✕ Cancel
          </button>
        )}
        {lastFile && !isProcessing && (
          <button
            className="btn bg-transparent border border-indigo-500 rounded-[4px] px-3.5 py-1.5 text-indigo-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-indigo-500/10"
            onClick={() => { pendingFileRef.current = lastFile; setShowModal(true) }}
          >
            ↺ Rescan
          </button>
        )}
      </div>
      <br />

      {/* Modal to be shown immediately after file drop */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal__title">Are there several people in this video?</div>
            <p className="modal__desc">
              Single person videos process faster and more accurately.<br />
              Choose multi-person to manually select who to track.
            </p>
            <div className="modal__actions">
              <button
                className="btn bg-transparent border border-[#39e8a0] rounded-[4px] px-5 py-2 text-[#39e8a0] text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-[#39e8a0]/10"
                onClick={handleSinglePerson}
              >
                Just one person
              </button>
              <button
                className="btn bg-transparent border border-[#7c6cff] rounded-[4px] px-5 py-2 text-[#7c6cff] text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-[#7c6cff]/10"
                onClick={handleMultiPerson}
              >
                Multiple people
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Output Banner */}
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
            <StatBox label="Sampled"   value={stats.totalSampled}  unit={`@ ${stats.captureFps}fps`} />
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
                  className="btn bg-transparent border border-[#f5a623] rounded-[4px] px-3.5 py-1.5 text-[#f5a623] text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-[#f5a623]/10"
                  onClick={() => exportBVH(frames, stats.captureFps)}
                  >
                  ↓ Export BVH
                </button>
                {backendAvailable === true && (
                <button
                  className="btn bg-transparent border border-purple-400 rounded-[4px] px-3.5 py-1.5 text-purple-400 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-purple-400/10"
                  onClick={exportEnhancedBVH}
                  >
                    ↓ Export Enhanced BVH ✦
                  </button>
                )}
                {backendAvailable === false && (
                  <span className="text-xs font-mono text-gray-500 self-center">
                    (Enhanced mode: start backend for ↑ quality)
                  </span>
                )}
                <button
                  className="btn bg-transparent border border-green-500 rounded-[4px] px-3.5 py-1.5 text-green-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-green-500/10"
                  onClick={exportJSON}
                  >
                  ↓ Export JSON
                </button>
              </div>
          </div>
          <div className="inspector-card">
            <FrameInspector frames={frames} stats={stats} />
          </div>
        </>
      )}

    </div>
  )
}