import { useRef, useState } from 'react'
import { usePoseExtractor, DEFAULT_SETTINGS } from './hooks/usePoseExtractor'
import FrameInspector from './components/FrameInspector'
import './App.css'

const STATUS_COLOR = {
  idle:            '#4a4a6a',
  'loading-model': '#f5a623',
  processing:      '#6366f1',
  done:            '#22c55e',
  error:           '#ef4444',
}

const STATUS_LABEL = {
  idle:            '● Idle',
  'loading-model': '◌ Loading model…',
  processing:      '◎ Processing…',
  done:            '✓ Done',
  error:           '✕ Error',
}

function Tag({ children, color = '#4a4a6a' }) {
  return (
    <span className="tag" style={{ color, borderColor: color }}>
      {children}
    </span>
  )
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
  const { processVideo, status, progress, frames, stats, error } = usePoseExtractor()
  const [dragOver, setDragOver] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [lastFile, setLastFile] = useState(null)
  const inputRef = useRef(null)

  function setSetting(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) return
      setLastFile(file)
    processVideo(file, settings)
    // Reset input so the same file can be re-selected
    if (inputRef.current) inputRef.current.value = ''
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
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

  const isProcessing = status === 'processing' || status === 'loading-model'

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
          {/* Tailwind: only additive here for the rose button */}
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
              value={settings.confidenceThreshold}
              onChange={(e) => setSetting('confidenceThreshold', Number(e.target.value))}
            />
            <span className="setting__hint">
              Drop frames where the positions of joint are uncertain — higher = fewer, more accurate frames
            </span>
          </div>

          <div className="setting">
            <div className="setting__header">
              <span className="setting__label">Keyframe sensitivity</span>
              <span className="setting__value">{settings.keyframeThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={0.2} step={0.01}
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
            <input type="range" min={10} max={200} step={5}
              value={settings.maxFrames}
              onChange={(e) => setSetting('maxFrames', Number(e.target.value))}
            />
            <span className="setting__hint">
              Hard frame cap — spreads frames evenly if total number of keyframes exceeds this cap
            </span>
          </div>

        </div>
      </div>

      {/* Upload zone */}
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
      <div className="upload-row flex items-center justify-between width-full">
        {/* Left */}
        <div></div>
        {/* Right */}
        {lastFile && !isProcessing && (
          <button
            className="btn bg-transparent border border-indigo-500 rounded-[4px] px-3.5 py-1.5 text-indigo-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-indigo-500/10 whitespace-nowrap"
            onClick={() => processVideo(lastFile, settings)}
          >
            ↺ Rescan
          </button>
        )}
      </div>
      
      {/* Error */}
      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Stats */}
      {stats && (
        <div className="stats-row">
          <StatBox label="Sampled"   value={stats.totalSampled}  unit={`@ ${stats.captureFps}fps`} />
          <StatBox label="Confident" value={stats.capturedCount} unit="frames" />
          <StatBox label="Keyframes" value={stats.keyframeCount} unit="unique" />
          <StatBox label="Final"     value={stats.frameCount}    unit="kept" />
        </div>
      )}

      {/* Frame inspector */}
      {frames.length > 0 && (
        <>
          <div className="inspector-header">
            <h2 className="inspector-header__title">Frame Inspector</h2>
            <button className="btn bg-transparent border border-green-500 rounded-[4px] px-3.5 py-1.5 text-green-500 text-xs font-mono cursor-pointer transition-colors duration-100 hover:bg-green-500/10" onClick={exportJSON}>
              ↓ Export JSON
            </button>
          </div>

          <div className="inspector-card">
            <FrameInspector frames={frames} stats={stats} />
          </div>
        </>
      )}

    </div>
  )
}