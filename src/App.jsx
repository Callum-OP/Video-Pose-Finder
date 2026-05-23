import { useRef, useState } from 'react'
import { usePoseExtractor, DEFAULT_SETTINGS } from './hooks/usePoseExtractor'
import FrameInspector from './components/FrameInspector'
import './App.css'

const STATUS_COLOR = {
  idle:            '#4a4a6a',
  'loading-model': '#f5a623',
  processing:      '#7c6cff',
  done:            '#39e8a0',
  error:           '#ff4d6d',
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
  const inputRef = useRef(null)

  function setSetting(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) return
    processVideo(file, settings)
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
    dragOver        ? 'upload-zone--drag' : '',
    isProcessing    ? 'upload-zone--busy' : '',
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
        <span
          className="status-bar__label"
          style={{ color: STATUS_COLOR[status] }}
        >
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
          <span className="settings__title">extraction settings</span>
          <button className=" border-rose-500 rounded text-rose-700" onClick={() => setSettings(DEFAULT_SETTINGS)}>
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
              How often to sample the video - lower = fewer duplicate poses in lower fps videos
            </span>
          </div>

          <div className="setting">
            <div className="setting__header">
              <span className="setting__label">confidence threshold</span>
              <span className="setting__value">{Math.round(settings.confidenceThreshold * 100)}%</span>
            </div>
            <input type="range" min={0.1} max={0.95} step={0.05}
              value={settings.confidenceThreshold}
              onChange={(e) => setSetting('confidenceThreshold', Number(e.target.value))}
            />
            <span className="setting__hint">
              Drop frames where MediaPipe is uncertain about joint positions
            </span>
          </div>

          <div className="setting">
            <div className="setting__header">
              <span className="setting__label">keyframe sensitivity</span>
              <span className="setting__value">{settings.keyframeThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={0.2} step={0.01}
              value={settings.keyframeThreshold}
              onChange={(e) => setSetting('keyframeThreshold', Number(e.target.value))}
            />
            <span className="setting__hint">
              Minimum pose difference to count as new keyframe - higher = fewer and more unique poses per frame
            </span>
          </div>

          <div className="setting">
            <div className="setting__header">
              <span className="setting__label">max frames</span>
              <span className="setting__value">{settings.maxFrames}</span>
            </div>
            <input type="range" min={10} max={200} step={5}
              value={settings.maxFrames}
              onChange={(e) => setSetting('maxFrames', Number(e.target.value))}
            />
            <span className="setting__hint">
              Hard frame cap -if keyframes exceed this then evenly subsample
            </span>
          </div>

        </div>
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
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="upload-zone__icon">⬆</div>
        <div className="upload-zone__label">
          {isProcessing ? 'Processing…' : 'Drop a video or click to upload'}
        </div>
        <div className="upload-zone__hint">
          MP4, MOV, WebM
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Stats */}
      {stats && (
        <div className="stats-row">
          <StatBox label="sampled"    value={stats.totalSampled}  unit={`@ ${stats.captureFps}fps`} />
          <StatBox label="confident"  value={stats.capturedCount} unit="frames" />
          <StatBox label="keyframes"  value={stats.keyframeCount} unit="unique" />
          <StatBox label="final"      value={stats.frameCount}    unit="kept" />
        </div>
      )}

      {/* Frame inspector */}
      {frames.length > 0 && (
        <>
          <div className="inspector-header">
            <h2 className="inspector-header__title">Frame Inspector</h2>
            <button className="btn-export" onClick={exportJSON}>
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