import { useCallback, useEffect, useState } from 'react'
import { Check, HardDrive, LoaderCircle, Trash2, X } from 'lucide-react'
import { Modal } from '@/components/Modal'
import { readableError } from '@/lib/errors'

// One row of GET /api/models/cache. sizeBytes is null when the model has no folder on disk.
export type CacheEntry = { id: string; cached: boolean; sizeBytes: number | null }
type CatalogModel = { id: string; name?: string }

const NOTICE_MS = 6000

// Sizes read like the model list: 1.6 GB, 500.0 MB, 512 KB.
export function fmtModelSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`
  return `${bytes} B`
}

// Model ids may contain a slash (owner/repository). Each segment is encoded so the route still matches.
const modelPath = (id: string) => id.split('/').map(encodeURIComponent).join('/')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    const detail = typeof data?.detail === 'string' ? data.detail : ''
    throw Object.assign(new Error(detail || `Request failed (${response.status})`), { status: response.status })
  }
  return response.json() as Promise<T>
}

export function ModelStorage({ models, online, busy, onChanged }: { models: CatalogModel[]; online: boolean; busy: boolean; onChanged: () => void }) {
  const [entries, setEntries] = useState<CacheEntry[]>([])
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [notice, setNotice] = useState('')
  const [target, setTarget] = useState<CacheEntry | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  const nameOf = (id: string) => models.find(model => model.id === id)?.name || id

  const loadEntries = useCallback(async () => {
    setLoading(true); setLoadError('')
    try {
      const data = await request<{ models: CacheEntry[] }>('/api/models/cache')
      setEntries(data.models); setLoaded(true)
    } catch (e) { setLoadError(readableError(e)) }
    finally { setLoading(false) }
  }, [])

  // The parent passes a new `models` list on each refresh, so a model downloaded or loaded elsewhere shows up here too.
  useEffect(() => { if (online) void loadEntries() }, [online, models, loadEntries])

  const flash = (message: string) => {
    setNotice(message)
    window.setTimeout(() => setNotice(''), NOTICE_MS)
  }

  const closeDelete = () => { if (!deleting) { setTarget(null); setDeleteError('') } }

  const confirmDelete = async () => {
    if (!target) return
    setDeleting(true); setDeleteError('')
    try {
      const result = await request<{ id: string; deleted: boolean; freedBytes: number }>(`/api/models/cache/${modelPath(target.id)}`, { method: 'DELETE' })
      setTarget(null)
      flash(`Freed ${fmtModelSize(result.freedBytes)}`)
      await loadEntries()
      onChanged()
    } catch (e) {
      // A 409 keeps the dialog open with the server's reason, so the user can cancel and try later.
      setDeleteError(readableError(e))
    } finally { setDeleting(false) }
  }

  const cachedTotal = entries.reduce((sum, entry) => sum + (entry.cached && entry.sizeBytes ? entry.sizeBytes : 0), 0)
  const targetName = target ? nameOf(target.id) : ''

  return <section className="panel model-storage-panel" aria-labelledby="model-storage-heading">
    <div className="panel-heading"><div className="step-icon"><HardDrive size={13}/></div><div><h2 id="model-storage-heading">Model storage</h2><p>Downloaded models take disk space on this Mac. Delete the ones you no longer need.</p></div></div>
    {notice && <div className="alert success-alert" role="status"><Check size={16}/>{notice}</div>}
    {loadError && <div className="alert error-alert" role="alert"><X size={16}/><div><strong>Could not read model storage</strong><span>{loadError}</span></div></div>}
    {!online
      ? <p className="field-hint">Start the local engine to see which models are downloaded.</p>
      : !loaded && loading
        ? <div className="project-loading"><LoaderCircle size={16} className="spin"/> Reading model storage…</div>
        : <>
          <ul className="model-storage-list" aria-label="Model storage list">
            {entries.map(entry => {
              const name = nameOf(entry.id)
              return <li key={entry.id} className="model-storage-item">
                <span className="model-storage-name" title={entry.id}>{name}</span>
                <span className="model-storage-size">{entry.cached ? (entry.sizeBytes === null ? 'Size unknown' : fmtModelSize(entry.sizeBytes)) : '—'}</span>
                <span className={`batch-pill ${entry.cached ? 'pill-completed' : ''}`}>{entry.cached ? 'Cached' : 'Not cached'}</span>
                {entry.cached
                  ? <button className="icon-button" onClick={() => { setDeleteError(''); setTarget(entry) }} disabled={busy} aria-label={`Delete ${name} files`} title="Delete downloaded files"><Trash2 size={16}/></button>
                  : <span className="model-storage-spacer" aria-hidden="true"/>}
              </li>
            })}
          </ul>
          {!entries.length && !loadError && <p className="field-hint">No models in the catalog.</p>}
          {entries.length > 0 && <div className="model-storage-foot"><span>{entries.filter(entry => entry.cached).length} of {entries.length} cached</span><span>{fmtModelSize(cachedTotal)} used</span></div>}
        </>}
    <Modal open={Boolean(target)} title="Delete downloaded model?" onClose={closeDelete}>
      <p className="dialog-copy">The downloaded files for “{targetName}” will be removed from this Mac. Other models are not affected. You can download it again later from Hugging Face.</p>
      {deleteError && <div className="dialog-error" role="alert">{deleteError}</div>}
      <div className="dialog-actions"><button className="button-secondary" onClick={closeDelete} disabled={deleting}>Cancel</button><button className="button-danger" onClick={() => void confirmDelete()} disabled={deleting}><Trash2 size={14}/>{deleting ? 'Deleting…' : 'Delete model files'}</button></div>
    </Modal>
  </section>
}
