import { ChangeEvent, useEffect, useRef, useState } from 'react'
import { AudioLines, Check, ChevronDown, Download, LoaderCircle, Trash2, Upload, X } from 'lucide-react'
import { SpeakerLabelOption, SpeakerSupport } from '@/components/SpeakerLabelOption'
import { Modal } from '@/components/Modal'
import { readableError } from '@/lib/errors'
import { fileAcceptAttribute, isAcceptedMediaFile, type MediaCaps } from '@/lib/mediaAccept'

// Whisper-only fields are optional: they are sent only when the selected runtime supports them.
export type BatchSettings = { model: string; runtime: string; compute: string; language: string; beam?: number; vad?: boolean; prompt?: string }

type BatchJob = { id: string; filename?: string; status: string; progress?: number | null; error?: string | null; segments?: Array<{ start: number; end: number; text: string; speaker?: string }>; info?: Record<string, unknown> | null }
type BatchState = { id: string; total: number; done: boolean; counts: Record<string, number>; jobs: BatchJob[] }
// One row of GET /api/batches. The list may omit per-file jobs, so only counts and totals are relied on.
type BatchSummary = { id: string; total?: number; done: boolean; counts?: Record<string, number>; created_at?: number }

const DEFAULT_MAX_FILES = 25
const HISTORY_PAGE = 20

function timecode(sec: number) { const s = Math.max(0, Math.floor(sec || 0)); return `${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` }

async function batchRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    let detail = ''
    try { const data = await response.json(); detail = data.detail || data.error || data.message || '' } catch { detail = await response.text() }
    throw Object.assign(new Error(detail || `Request failed (${response.status})`), { status: response.status })
  }
  try {
    return await response.json() as T
  } catch {
    throw new Error('The local service sent a reply that could not be read.')
  }
}

function parseHistory(data: unknown): { batches: BatchSummary[]; total: number } {
  const list = Array.isArray(data) ? data : (data as { batches?: unknown })?.batches
  if (!Array.isArray(list)) throw new Error('The batch list could not be read.')
  const total = Array.isArray(data) ? list.length : Number((data as { total?: unknown }).total ?? list.length)
  return { batches: list as BatchSummary[], total }
}

function batchStatus(item: BatchSummary): { key: string; label: string } {
  if (!item.done) return { key: 'running', label: 'Running' }
  if ((item.counts?.failed ?? 0) > 0) return { key: 'failed', label: 'Finished with failures' }
  if ((item.counts?.cancelled ?? 0) > 0) return { key: 'cancelled', label: 'Cancelled' }
  return { key: 'completed', label: 'Completed' }
}

function fmtBatchDate(seconds?: number) {
  return typeof seconds === 'number' ? new Date(seconds * 1000).toLocaleString('en-AU', { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit' }) : ''
}

const STATUS_LABEL: Record<string, string> = { queued: 'Waiting', running: 'Transcribing', completed: 'Done', failed: 'Failed', cancelled: 'Cancelled' }

export function BatchPanel({ settings, online, disabled, speakerSupport, labelSpeakers, onLabelSpeakersChange, onRunningChange, mediaCaps, maxBatchFiles = DEFAULT_MAX_FILES }: { settings: BatchSettings; online: boolean; disabled: boolean; speakerSupport: SpeakerSupport; labelSpeakers: boolean; onLabelSpeakersChange: (value: boolean) => void; onRunningChange?: (running: boolean) => void; mediaCaps?: MediaCaps | null; maxBatchFiles?: number }) {
  const maxFiles = maxBatchFiles > 0 ? maxBatchFiles : DEFAULT_MAX_FILES
  const inputRef = useRef<HTMLInputElement>(null)
  const retryInputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [batch, setBatch] = useState<BatchState | null>(null)
  // The files of the current batch, aligned with batch.jobs. They live only in memory, so a reload drops them.
  const [batchFiles, setBatchFiles] = useState<Array<File | undefined>>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [history, setHistory] = useState<BatchSummary[]>([])
  const [historyTotal, setHistoryTotal] = useState(0)
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [historyNotice, setHistoryNotice] = useState('')
  const [openingId, setOpeningId] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<BatchSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  const running = Boolean(batch && !batch.done)
  const failedIndexes = batch?.done ? batch.jobs.flatMap((job, index) => job.status === 'failed' ? [index] : []) : []
  const heldFile = (index: number) => {
    const file = batchFiles[index]
    const name = batch?.jobs[index]?.filename
    return file && (!name || file.name.slice(0, 255) === name) ? file : undefined
  }
  const allFailedHeld = failedIndexes.length > 0 && failedIndexes.every(index => Boolean(heldFile(index)))

  useEffect(() => { onRunningChange?.(running) }, [running])

  useEffect(() => {
    if (!batch || batch.done) return
    let cancelled = false
    const poll = async () => {
      try {
        const next = await batchRequest<BatchState>(`/api/batches/${encodeURIComponent(batch.id)}`)
        if (!cancelled) { setBatch(next); setError('') }
      } catch (e) {
        if (!cancelled) setError(readableError(e, 'Could not read the batch status.'))
      }
    }
    const timer = window.setInterval(poll, 1200)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [batch?.id, batch?.done])

  const loadHistory = async (offset = 0) => {
    setHistoryLoading(true); setHistoryError('')
    try {
      const page = parseHistory(await batchRequest<unknown>(`/api/batches?limit=${HISTORY_PAGE}&offset=${offset}`))
      setHistory(current => offset ? [...current, ...page.batches.filter(item => !current.some(known => known.id === item.id))] : page.batches)
      setHistoryTotal(page.total); setHistoryLoaded(true)
    } catch (e) {
      setHistoryError(readableError(e, 'Could not load batch history.'))
    } finally {
      setHistoryLoading(false)
    }
  }

  // Refresh the list on mount and whenever a batch finishes, so new batches appear without a reload.
  useEffect(() => { if (online) void loadHistory(0) }, [online, batch?.id, batch?.done])

  const openBatch = async (item: BatchSummary) => {
    setOpeningId(item.id); setError(''); setHistoryNotice('')
    try {
      const next = await batchRequest<BatchState>(`/api/batches/${encodeURIComponent(item.id)}`)
      // Files are not kept for batches opened from history, so a failed batch asks for the files again.
      setBatch(next); setBatchFiles([]); setExpanded(null)
    } catch (e) {
      setError(readableError(e, 'Could not open this batch.'))
    } finally {
      setOpeningId('')
    }
  }

  const closeDelete = () => { if (!deleting) { setDeleteTarget(null); setDeleteError('') } }

  const confirmDelete = async () => {
    if (!deleteTarget) return
    const removed = deleteTarget
    setDeleting(true); setDeleteError('')
    try {
      await batchRequest(`/api/batches/${encodeURIComponent(removed.id)}`, { method: 'DELETE' })
      setDeleteTarget(null)
      setHistory(current => current.filter(item => item.id !== removed.id)); setHistoryTotal(total => Math.max(0, total - 1))
      if (batch?.id === removed.id) { setBatch(null); setBatchFiles([]) }
      setHistoryNotice('Batch deleted from this Mac.')
      window.setTimeout(() => setHistoryNotice(''), 4000)
    } catch (e) {
      if ((e as { status?: number }).status === 404) {
        setDeleteTarget(null)
        setHistory(current => current.filter(item => item.id !== removed.id))
        setHistoryNotice('That batch was already deleted.')
      } else {
        setDeleteError(readableError(e, 'The batch could not be deleted.'))
      }
    } finally {
      setDeleting(false)
    }
  }

  const addFiles = (picked: FileList | null) => {
    setError('')
    if (!picked) return
    const accepted: File[] = []
    const rejected: File[] = []
    Array.from(picked).forEach(file => (isAcceptedMediaFile(file, mediaCaps) ? accepted : rejected).push(file))
    if (rejected.length) setError(`Skipped ${rejected.length} file${rejected.length === 1 ? '' : 's'} that are not audio or video: ${rejected.slice(0, 3).map(f => f.name).join(', ')}`)
    setFiles(current => {
      const merged = [...current, ...accepted].slice(0, maxFiles)
      if (current.length + accepted.length > maxFiles) setError(`A batch can hold at most ${maxFiles} files.`)
      return merged
    })
  }

  const submit = async (picked: File[]) => {
    if (!picked.length || submitting || disabled) return
    setSubmitting(true); setError('')
    const body = new FormData()
    picked.forEach(file => body.append('files', file, file.name))
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
      setBatchFiles(picked)
      setFiles([])
      setExpanded(null)
    } catch (e) {
      setError(readableError(e, 'The batch could not be started.'))
    } finally {
      setSubmitting(false)
    }
  }

  const start = () => submit(files)

  // Retry submits only the failed files, in their original order, when all of them are still in memory.
  const retryFailed = () => {
    if (!batch || !allFailedHeld) return
    const held = failedIndexes.flatMap(index => { const file = heldFile(index); return file ? [file] : [] })
    void submit(held)
  }

  // After a reload the files are gone: the user picks the failed files again, matched by name.
  const attachFailed = (picked: FileList | null) => {
    if (!batch || !picked) return
    const next = batch.jobs.map((_, index) => heldFile(index))
    const unmatched: string[] = []
    Array.from(picked).forEach(file => {
      const index = failedIndexes.find(i => !next[i] && batch.jobs[i].filename === file.name)
      if (index === undefined) unmatched.push(file.name)
      else next[index] = file
    })
    setBatchFiles(next)
    setError(unmatched.length ? `These files are not among the failed files: ${unmatched.slice(0, 3).join(', ')}` : '')
  }

  const cancelBatch = async () => {
    if (!batch) return
    try {
      setBatch(await batchRequest<BatchState>(`/api/batches/${encodeURIComponent(batch.id)}/cancel`, { method: 'POST' }))
    } catch (e) {
      setError(readableError(e, 'Could not cancel the batch.'))
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
            <span className="field-hint">Up to {maxFiles} files. Processed in order on this Mac.</span>
            <input ref={inputRef} type="file" multiple accept={fileAcceptAttribute(mediaCaps)} hidden aria-label="Batch audio files"
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
            <button type="button" className="button-primary" onClick={() => void start()} disabled={!files.length || submitting || disabled || !online}>
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
            {batch.done && failedIndexes.length > 0 && (allFailedHeld
              ? <button type="button" className="button-primary" onClick={retryFailed} disabled={submitting || disabled}><AudioLines size={14}/> Retry failed ({failedIndexes.length})</button>
              : <>
                  <p className="field-hint">Select the failed files again to retry.</p>
                  <button type="button" className="button-secondary" onClick={() => retryInputRef.current?.click()} disabled={submitting || disabled}><Upload size={13}/> Choose failed files</button>
                  <input ref={retryInputRef} type="file" multiple accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.wma,.mp4,.webm" hidden aria-label="Failed batch files"
                    onChange={(e: ChangeEvent<HTMLInputElement>) => { attachFailed(e.target.files); e.target.value = '' }}/>
                </>)}
            {batch.done && <button type="button" className="button-primary batch-new" onClick={() => { setBatch(null); setBatchFiles([]); setError('') }}>New batch</button>}
          </div>
          {running && <p className="field-hint">Keep this window open. Files finish one at a time.</p>}
        </div>
      )}

      {error && <div className="alert error-alert" role="alert"><X size={15}/><div><span>{error}</span></div></div>}

      <div className="batch-history">
        <h3>Batch history</h3>
        {historyNotice && <p className="field-hint" role="status">{historyNotice}</p>}
        {!online ? <p className="field-hint">The local engine is offline, so batch history cannot be read.</p> : <>
          {historyError && <div className="batch-history-error"><span>{historyError}</span><button type="button" className="button-secondary" onClick={() => void loadHistory(0)} disabled={historyLoading}>Try again</button></div>}
          {!historyLoaded && historyLoading && !historyError && <div className="project-loading"><LoaderCircle size={16} className="spin"/> Reading batches…</div>}
          {historyLoaded && !history.length && !historyError && <div className="history-state"><h3>No batches yet.</h3><p>Batches you start are kept here until you delete them.</p></div>}
          {history.length > 0 && <>
            <ol className="batch-list" aria-label="Batch history list">
              {history.map(item => {
                const status = batchStatus(item)
                const failed = item.counts?.failed ?? 0
                const files = item.total ?? 0
                return <li key={item.id} className={`batch-item status-${status.key}`}>
                  <div className="batch-item-row">
                    <span className="batch-name">{files} {files === 1 ? 'file' : 'files'}{item.created_at ? ` · ${fmtBatchDate(item.created_at)}` : ''}</span>
                    <span className={`batch-pill pill-${status.key}`}>{status.label}</span>
                  </div>
                  <div className="batch-actions">
                    <button type="button" className="button-secondary" onClick={() => void openBatch(item)} disabled={running || Boolean(openingId)}>
                      {openingId === item.id ? <LoaderCircle size={13} className="spin"/> : null} Open{failed ? ` · ${failed} failed` : ''}
                    </button>
                    <button type="button" className="button-danger" onClick={() => { setDeleteError(''); setDeleteTarget(item) }} disabled={!item.done || deleting} title={item.done ? 'Delete this batch and its transcripts' : 'Cancel the batch before deleting it'}>
                      <Trash2 size={13}/> Delete
                    </button>
                  </div>
                </li>
              })}
            </ol>
            <div className="history-foot"><span>Showing {history.length} of {historyTotal}</span>{history.length < historyTotal && <button type="button" className="button-secondary" onClick={() => void loadHistory(history.length)} disabled={historyLoading}>{historyLoading ? 'Loading…' : 'Load more'}</button>}</div>
          </>}
        </>}
      </div>

      <Modal open={Boolean(deleteTarget)} title="Delete batch?" onClose={closeDelete}>
        <p className="dialog-copy">This batch and its transcripts will be removed from this Mac. Audio was never stored, so this cannot be undone.</p>
        {deleteError && <div className="dialog-error" role="alert">{deleteError}</div>}
        <div className="dialog-actions"><button className="button-secondary" onClick={closeDelete} disabled={deleting}>Cancel</button><button className="button-danger" onClick={() => void confirmDelete()} disabled={deleting}><Trash2 size={14}/>{deleting ? 'Deleting…' : 'Delete batch'}</button></div>
      </Modal>
    </section>
  )
}
