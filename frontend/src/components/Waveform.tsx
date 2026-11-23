import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

const BINS = 600
// Decoding holds the whole file as PCM in memory, so very large recordings show "Waveform unavailable" instead.
const MAX_DECODE_BYTES = 100 * 1024 * 1024

type Span = { start: number; end: number }
type Props = { file: File; segments: Span[]; duration: number; currentTime: number; onSeek: (time: number) => void }
type Decoded = { bins: Float32Array; duration: number }

// Reduces raw samples to one peak (the largest absolute value) per bin, so drawing cost does not depend on file length.
export function summarizePeaks(samples: Float32Array, bins: number): Float32Array {
  const out = new Float32Array(bins)
  const step = samples.length / bins
  for (let i = 0; i < bins; i++) {
    const from = Math.floor(i * step)
    const to = Math.max(from + 1, Math.floor((i + 1) * step))
    let peak = 0
    for (let j = from; j < to && j < samples.length; j++) {
      const value = Math.abs(samples[j])
      if (value > peak) peak = value
    }
    out[i] = peak
  }
  return out
}

// Waveform timeline for the loaded audio. The audio element stays the source of truth: this component only draws and seeks.
export function Waveform({ file, segments, duration, currentTime, onSeek }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dragging = useRef(false)
  const [decoded, setDecoded] = useState<Decoded | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'failed'>('loading')
  const [resizeTick, setResizeTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    setDecoded(null)
    setState('loading')
    const AudioContextCtor: typeof AudioContext | undefined = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (file.size > MAX_DECODE_BYTES || !AudioContextCtor) { setState('failed'); return }
    const context = new AudioContextCtor()
    file.arrayBuffer()
      .then(data => context.decodeAudioData(data))
      .then(buffer => {
        if (cancelled) return
        setDecoded({ bins: summarizePeaks(buffer.getChannelData(0), BINS), duration: buffer.duration })
        setState('ready')
      })
      .catch(() => { if (!cancelled) setState('failed') })
      .finally(() => { void context.close().catch(() => {}) })
    return () => { cancelled = true }
  }, [file])

  useEffect(() => {
    const onResize = () => setResizeTick(tick => tick + 1)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !decoded) return
    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth || 600
    const height = canvas.clientHeight || 64
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr)
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const total = duration || decoded.duration || 1
    const playedX = Math.min(1, currentTime / total) * width
    const barWidth = width / BINS
    const middle = height / 2
    const ink = getComputedStyle(canvas).color || '#333'
    ctx.fillStyle = ink
    decoded.bins.forEach((peak, i) => {
      const x = i * barWidth
      const barHeight = Math.max(1, peak * height * 0.9)
      ctx.globalAlpha = x < playedX ? 0.9 : 0.35
      ctx.fillRect(x + barWidth * 0.15, middle - barHeight / 2, Math.max(1, barWidth * 0.7), barHeight)
    })
    ctx.globalAlpha = 1
    // Segment boundaries: the start of each segment, as a thin marker.
    ctx.fillStyle = 'rgba(128, 128, 128, 0.85)'
    for (const segment of segments) ctx.fillRect(Math.min(width - 1, (segment.start / total) * width), 0, 1, height)
    // Playhead.
    ctx.fillStyle = '#d1495b'
    ctx.fillRect(Math.min(width - 2, playedX), 0, 2, height)
  }, [decoded, segments, duration, currentTime, resizeTick])

  const seekFromPointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    if (rect.width <= 0) return
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width))
    const total = duration || decoded?.duration || 0
    if (total > 0) onSeek(ratio * total)
  }

  if (state === 'failed') {
    return <div className="waveform-timeline" data-state="failed"><p className="field-hint">Waveform unavailable</p></div>
  }
  if (state === 'loading') {
    return <div className="waveform-timeline" data-state="loading"><p className="field-hint">Reading waveform…</p></div>
  }
  return <div className="waveform-timeline" data-state="ready">
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Waveform timeline. Click or drag to seek."
      style={{ width: '100%', height: 64, display: 'block', cursor: 'pointer', touchAction: 'none' }}
      onPointerDown={event => { dragging.current = true; event.currentTarget.setPointerCapture(event.pointerId); seekFromPointer(event) }}
      onPointerMove={event => { if (dragging.current) seekFromPointer(event) }}
      onPointerUp={event => { dragging.current = false; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId) }}
      onPointerCancel={() => { dragging.current = false }}
    />
  </div>
}
