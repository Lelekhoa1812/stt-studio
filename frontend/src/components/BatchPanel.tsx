import { ChangeEvent, useEffect, useRef, useState } from 'react'
import { AudioLines, Check, ChevronDown, Download, LoaderCircle, Upload, X } from 'lucide-react'
import { SpeakerLabelOption, SpeakerSupport } from '@/components/SpeakerLabelOption'

// Whisper-only fields are optional: they are sent only when the selected runtime supports them.
export type BatchSettings = { model: string; runtime: string; compute: string; language: string; beam?: number; vad?: boolean; prompt?: string }

type BatchJob = { id: string; filename?: string; status: string; progress?: number | null; error?: string | null; segments?: Array<{ start: number; end: number; text: string; speaker?: string }>; info?: Record<string, unknown> | null }
type BatchState = { id: string; total: number; done: boolean; counts: Record<string, number>; jobs: BatchJob[] }

const ACCEPTED = /\.(wav|mp3|m4a|aac|flac|ogg|opus|wma|mp4|webm)$/i
const MAX_FILES = 25

function timecode(sec: number) { const s = Math.max(0, Math.floor(sec || 0)); return `${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` }

async function batchRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    let detail = ''
    try { const data = await response.json(); detail = data.detail || data.error || data.message || '' } catch { detail = await response.text() }
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

const STATUS_LABEL: Record<string, string> = { queued: 'Waiting', running: 'Transcribing', completed: 'Done', failed: 'Failed', cancelled: 'Cancelled' }

export function BatchPanel({ settings, online, disabled, speakerSupport, labelSpeakers, onLabelSpeakersChange }: { settings: BatchSettings; online: boolean; disabled: boolean; speakerSupport: SpeakerSupport; labelSpeakers: boolean; onLabelSpeakersChange: (value: boolean) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [batch, setBatch] = useState<BatchState | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const running = Boolean(batch && !batch.done)

  useEffect(() => {
    if (!batch || batch.done) return
    let cancelled = false
    const poll = async () => {
      try {
        const next = await batchRequest<BatchState>(`/api/batches/${encodeURIComponent(batch.id)}`)
        if (!cancelled) setBatch(next)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not read the batch status.')
      }
    }
    const timer = window.setInterval(poll, 1200)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [batch?.id, batch?.done])

  const addFiles = (picked: FileList | null) => {
    setError('')
    if (!picked) return
    const accepted: File[] = []
    const rejected: File[] = []
    Array.from(picked).forEach(file => (file.type.startsWith('audio/') || ACCEPTED.test(file.name) ? accepted : rejected).push(file))
    if (rejected.length) setError(`Skipped ${rejected.length} file${rejected.length === 1 ? '' : 's'} that are not audio: ${rejected.slice(0, 3).map(f => f.name).join(', ')}`)
    setFiles(current => {
      const merged = [...current, ...accepted].slice(0, MAX_FILES)
      if (current.length + accepted.length > MAX_FILES) setError(`A batch can hold at most ${MAX_FILES} files.`)
      return merged
    })
  }

  const start = async () => {
    if (!files.length || submitting || disabled) return
    setSubmitting(true); setError('')
    const body = new FormData()
    files.forEach(file => body.append('files', file, file.name))
    body.append('model', settings.model)
    body.append('runtime', settings.runtime)
    body.append('compute_type', settings.compute)
    if (settings.language !== 'auto') body.append('language', settings.language)
    if (settings.beam !== undefined) body.append('beam_size', String(settings.beam))
    if (settings.vad !== undefined) body.append('vad_filter', String(settings.vad))
    if (settings.prompt !== undefined) body.append('initial_prompt', settings.prompt)
    body.append('diarize', String(speakerSupport.available && labelSpeakers))
    try {
      const created = await batchRequest<BatchState>('/api/batches', { method: 'POST', body })
      setBatch(created)
      setFiles([])
      setExpanded(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The batch could not be started.')
    } finally {
      setSubmitting(false)
    }
  }

  const cancelBatch = async () => {
    if (!batch) return
    try {
      setBatch(await batchRequest<BatchState>(`/api/batches/${encodeURIComponent(batch.id)}/cancel`, { method: 'POST' }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not cancel the batch.')
    }
  }

  const downloadResults = () => {
    if (!batch) return
    const payload = {
      format: 'stt-studio-batch-results', version: 1, batch_id: batch.id, exported_at: new Date().toISOString(),
      files: batch.jobs.map(job => ({ filename: job.filename ?? null, status: job.status, error: job.error ?? null, info: job.info ?? null, segments: job.segments ?? [] })),
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = `batch-results-${batch.id.slice(0, 8)}.json`
    document.body.appendChild(link); link.click(); link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const completed = batch?.counts.completed ?? 0

  return (
    <section className="panel batch-panel" aria-labelledby="batch-heading">
      <div className="panel-heading">
        <div className="step-icon"><AudioLines size={13}/></div>
        <div>
          <h2 id="batch-heading">Batch queue</h2>
          <p>Transcribe several files one after another with the settings above. Each file keeps its own result.</p>
        </div>
      </div>

      {!batch && (
        <>
          <SpeakerLabelOption checked={labelSpeakers} support={speakerSupport} disabled={disabled || submitting} onChange={onLabelSpeakersChange}/>
          <div className="batch-picker">
            <button type="button" className="button-secondary" onClick={() => inputRef.current?.click()} disabled={disabled || submitting || !online} aria-label="Choose audio files for the batch">
              <Upload size={14}/> Add audio files
            </button>
            <span className="field-hint">Up to {MAX_FILES} files. Processed in order on this Mac.</span>
            <input ref={inputRef} type="file" multiple accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.wma,.mp4,.webm" hidden aria-label="Batch audio files"
              onChange={(e: ChangeEvent<HTMLInputElement>) => { addFiles(e.target.files); e.target.value = '' }}/>
          </div>
          {files.length > 0 && (
            <ul className="batch-files" aria-label="Files queued for the batch">
              {files.map((file, index) => (
                <li key={`${file.name}-${index}`}>
                  <span title={file.name}>{file.name}</span>
                  <button type="button" className="icon-button" onClick={() => setFiles(current => current.filter((_, i) => i !== index))} aria-label={`Remove ${file.name} from the batch`}><X size={13}/></button>
                </li>
              ))}
            </ul>
          )}
          <div className="run-row">
            <div className="run-note"><Check size={14}/><span>{files.length ? `${files.length} file${files.length === 1 ? '' : 's'} ready` : 'No files selected'}</span></div>
            <button type="button" className="button-primary" onClick={start} disabled={!files.length || submitting || disabled || !online}>
              {submitting ? <LoaderCircle size={16} className="spin"/> : <AudioLines size={16}/>}
              {submitting ? 'Starting…' : `Start batch${files.length ? ` (${files.length})` : ''}`}
            </button>
          </div>
        </>
      )}

      {batch && (
        <div className="batch-status">
          <div className="batch-summary" role="status">
            <strong>{completed} of {batch.total} complete</strong>
            <span>{batch.counts.running ? `${batch.counts.running} running · ` : ''}{batch.counts.queued ? `${batch.counts.queued} waiting` : batch.done ? 'finished' : ''}</span>
          </div>
          <ol className="batch-list">
            {batch.jobs.map(job => {
              const open = expanded === job.id
              const text = (job.segments ?? []).map(s => s.text).join(' ').trim()
              const speakers = [...new Set((job.segments ?? []).flatMap(s => s.speaker ? [s.speaker] : []))]
              const labelling = job.info?.diarization as { status?: string; reason?: string } | undefined
              return (
                <li key={job.id} className={`batch-item status-${job.status}`}>
                  <div className="batch-item-row">
                    <span className="batch-name" title={job.filename}>{job.filename ?? 'audio'}</span>
                    <span className={`batch-pill pill-${job.status}`}>{job.status === 'running' && <LoaderCircle size={11} className="spin"/>} {STATUS_LABEL[job.status] ?? job.status}{job.status === 'running' && typeof job.progress === 'number' ? ` · ${job.progress}%` : ''}</span>
                  </div>
                  {job.status === 'running' && typeof job.progress === 'number' && <div className="progress-bar"><span style={{ width: `${job.progress}%` }}/></div>}
                  {job.error && <p className="batch-error">{job.error}</p>}
                  {labelling?.status === 'failed' && (
                    <div className="speaker-failure"><strong>Speaker labels could not be produced.</strong><span>{labelling.reason || 'No reason was given.'}</span><span>The transcript is still usable.</span></div>
                  )}
                  {job.status === 'completed' && (
                    <button type="button" className="batch-toggle" onClick={() => setExpanded(open ? null : job.id)} aria-expanded={open}>
                      <ChevronDown size={13} className={open ? 'chev-open' : ''}/> {(job.segments ?? []).length} segments{open ? ' · hide text' : ' · show text'}
                    </button>
                  )}
                  {open && job.status === 'completed' && (
                    <div className="batch-text">
                      {(job.segments ?? []).map((s, i) => <p key={i}><span>{timecode(s.start)}</span>{s.speaker && <span className={`speaker-tag speaker-tone-${speakers.indexOf(s.speaker) % 4}`} title={s.speaker}>{s.speaker}</span>} {s.text}</p>)}
                      {!text && <p>No speech was detected in this file.</p>}
                    </div>
                  )}
                </li>
              )
            })}
          </ol>
          <div className="batch-actions">
            {!batch.done && <button type="button" className="button-secondary" onClick={cancelBatch}><X size={13}/> Cancel remaining</button>}
            {(completed > 0) && <button type="button" className="button-secondary" onClick={downloadResults}><Download size={13}/> Download results (JSON)</button>}
            {batch.done && <button type="button" className="button-primary batch-new" onClick={() => { setBatch(null); setError('') }}>New batch</button>}
          </div>
          {running && <p className="field-hint">Keep this window open. Files finish one at a time.</p>}
        </div>
      )}

      {error && <div className="alert error-alert" role="alert"><X size={15}/><div><span>{error}</span></div></div>}
    </section>
  )
}
