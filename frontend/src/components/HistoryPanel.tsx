import { useEffect, useState } from 'react'
import { AudioLines, Check, HardDrive, History, LoaderCircle, Trash2, X } from 'lucide-react'
import { Modal } from '@/components/Modal'

type HistoryItem = { id: string; status: string; created_at: number; model: string; language: string | null; filename: string | null; batch_id: string | null; info: Record<string, unknown> | null }
type StorageSummary = { jobs: number; batches: number; bytes: number }
type HistoryPage = { jobs: HistoryItem[]; total: number }

const PAGE_SIZE = 50
const STATUS_LABEL: Record<string, string> = { queued: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled' }
const OFFLINE_MESSAGE = 'The local engine is offline, so saved transcripts cannot be read. Start the local service, then reconnect from the sidebar.'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    const detail = typeof data?.detail === 'string' ? data.detail : ''
    throw Object.assign(new Error(detail || `Request failed (${response.status})`), { status: response.status })
  }
  return response.json() as Promise<T>
}

function readable(error: unknown) {
  if (error instanceof TypeError) return 'Can’t reach the local transcription service. Check that the STT backend is running, then retry.'
  return error instanceof Error ? error.message : 'Something went wrong.'
}

function statusOf(error: unknown) {
  return (error as { status?: number }).status
}

function fmtBytes(n: number) { if (n < 1024 ** 2) return `${Math.round(n / 1024)} KB`; return `${(n / 1024 ** 2).toFixed(1)} MB` }
function fmtDate(seconds: number) { return new Date(seconds * 1000).toLocaleString('en-AU', { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit' }) }

export function HistoryPanel({ online, activeId, languageName, onOpen, onDeleted }: { online: boolean; activeId?: string; languageName: (code: string) => string; onOpen: (job: Record<string, unknown>) => void; onDeleted: (id: string) => void }) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [total, setTotal] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [storage, setStorage] = useState<StorageSummary | null>(null)
  const [storageError, setStorageError] = useState(false)
  const [openingId, setOpeningId] = useState('')
  const [openError, setOpenError] = useState('')
  const [notice, setNotice] = useState('')
  const [target, setTarget] = useState<HistoryItem | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  const flash = (message: string) => { setNotice(message); window.setTimeout(() => setNotice(''), 4000) }

  const loadStorage = async () => {
    try {
      const data = await request<StorageSummary>('/api/storage')
      setStorage({ jobs: data.jobs, batches: data.batches, bytes: data.bytes }); setStorageError(false)
    } catch { setStorage(null); setStorageError(true) }
  }

  const loadFirstPage = async () => {
    setLoading(true); setLoadError('')
    try {
      const data = await request<HistoryPage>(`/api/jobs?limit=${PAGE_SIZE}&offset=0`)
      setItems(data.jobs); setTotal(data.total); setLoaded(true)
    } catch (e) { setLoadError(readable(e)); setItems([]); setTotal(0) }
    finally { setLoading(false) }
    void loadStorage()
  }

  const loadMore = async () => {
    setLoading(true); setLoadError('')
    try {
      const data = await request<HistoryPage>(`/api/jobs?limit=${PAGE_SIZE}&offset=${items.length}`)
      setItems(current => [...current, ...data.jobs.filter(job => !current.some(item => item.id === job.id))]); setTotal(data.total)
    } catch (e) { setLoadError(readable(e)) }
    finally { setLoading(false) }
  }

  useEffect(() => { if (online) void loadFirstPage() }, [online])

  const openItem = async (item: HistoryItem) => {
    setOpeningId(item.id); setOpenError(''); setNotice('')
    try { onOpen(await request<Record<string, unknown>>(`/api/jobs/${encodeURIComponent(item.id)}`)) }
    catch (e) {
      if (statusOf(e) === 404) { setOpenError('This transcript is no longer saved. The list has been refreshed.'); void loadFirstPage() }
      else setOpenError(readable(e))
    } finally { setOpeningId('') }
  }

  const closeDelete = () => { if (!deleting) { setTarget(null); setDeleteError('') } }

  const confirmDelete = async () => {
    if (!target) return
    const removed = target
    setDeleting(true); setDeleteError('')
    try {
      await request(`/api/jobs/${encodeURIComponent(removed.id)}`, { method: 'DELETE' })
      setTarget(null); onDeleted(removed.id); flash('Saved transcript deleted from this Mac.'); void loadFirstPage()
    } catch (e) {
      if (statusOf(e) === 404) { setTarget(null); onDeleted(removed.id); flash('That transcript was already deleted.'); void loadFirstPage() }
      else setDeleteError(readable(e))
    } finally { setDeleting(false) }
  }

  const name = (item: HistoryItem) => item.filename || 'Untitled audio'

  return <section className="panel history-panel" aria-labelledby="history-heading">
    <div className="panel-heading"><div className="step-icon"><History size={13}/></div><div><h2 id="history-heading">Saved transcripts</h2><p>Kept on this Mac until you delete them.</p></div></div>
    <p className="history-note"><HardDrive size={13}/> Audio is never stored, so saved transcripts cannot be played back.</p>
    {storage && <p className="history-storage" aria-label="Storage summary"><span>{storage.jobs} {storage.jobs === 1 ? 'transcript' : 'transcripts'}</span><i/><span>{storage.batches} {storage.batches === 1 ? 'batch' : 'batches'}</span><i/><span>{fmtBytes(storage.bytes)} used</span></p>}
    {storageError && <p className="field-hint">Storage summary unavailable.</p>}
    {notice && <div className="alert success-alert" role="status"><Check size={16}/>{notice}</div>}
    {openError && <div className="alert error-alert" role="alert"><X size={16}/><div><strong>Could not open this transcript</strong><span>{openError}</span></div></div>}
    {!online ? <div className="history-state"><div className="empty-orb warning-orb"><X size={21}/></div><h3>Local engine offline</h3><p>{OFFLINE_MESSAGE}</p></div> : <>
      {loadError && <div className="alert error-alert" role="alert"><X size={16}/><div><strong>Could not load saved transcripts</strong><span>{loadError}</span></div><button className="button-secondary" onClick={() => void (items.length ? loadMore() : loadFirstPage())} disabled={loading}>Try again</button></div>}
      {!loaded && loading && !loadError && <div className="project-loading"><LoaderCircle size={16} className="spin"/> Reading saved transcripts…</div>}
      {loaded && !items.length && !loadError && <div className="history-state"><div className="empty-orb"><AudioLines size={22}/></div><h3>No saved transcripts yet.</h3><p>Transcribe a file and it will be kept here until you delete it.</p></div>}
      {items.length > 0 && <>
        <ol className="history-list" aria-label="Saved transcripts list">
          {items.map(item => {
            const code = (item.info?.language as string | undefined) || item.language
            return <li key={item.id} className={`history-item${item.id === activeId ? ' is-active' : ''}`}>
              <button className="history-open" onClick={() => void openItem(item)} disabled={Boolean(openingId)} aria-current={item.id === activeId || undefined}>
                <span className="history-name" title={name(item)}>{name(item)}</span>
                <span className="history-meta"><span className={`batch-pill pill-${item.status}`}>{STATUS_LABEL[item.status] ?? item.status}</span><span>{fmtDate(item.created_at)}</span><span>{item.model}</span><span>{code ? languageName(code) : 'Language not recorded'}</span>{item.batch_id && <span className="history-batch">In batch</span>}{openingId === item.id && <LoaderCircle size={11} className="spin"/>}</span>
              </button>
              <button className="icon-button" onClick={() => { setDeleteError(''); setTarget(item) }} aria-label={`Delete ${name(item)}`} title="Delete transcript"><Trash2 size={16}/></button>
            </li>
          })}
        </ol>
        <div className="history-foot"><span>Showing {items.length} of {total}</span>{items.length < total && <button className="button-secondary" onClick={() => void loadMore()} disabled={loading}>{loading ? 'Loading…' : 'Load more'}</button>}</div>
      </>}
    </>}
    <Modal open={Boolean(target)} title="Delete saved transcript?" onClose={closeDelete}>
      <p className="dialog-copy">“{target ? name(target) : ''}” and its transcript and settings will be removed from this Mac. Audio was never stored, so this cannot be undone.{target?.batch_id ? ' It belongs to a batch: only this file is removed, and the batch keeps its other files.' : ''}</p>
      {deleteError && <div className="dialog-error" role="alert">{deleteError}</div>}
      <div className="dialog-actions"><button className="button-secondary" onClick={closeDelete} disabled={deleting}>Cancel</button><button className="button-danger" onClick={() => void confirmDelete()} disabled={deleting}><Trash2 size={14}/>{deleting ? 'Deleting…' : 'Delete transcript'}</button></div>
    </Modal>
  </section>
}
