import { ChangeEvent, DragEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Switch from '@radix-ui/react-switch'
import { Button } from '@/components/ui/button'
import { BatchPanel } from '@/components/BatchPanel'
import { SpeakerLabelOption, SpeakerSupport, UNKNOWN_SPEAKER_SUPPORT } from '@/components/SpeakerLabelOption'
import { MAX_SPEAKER_NAME, PROJECT_FILE_EXTENSION, ProjectFileError, decodeProjectFile, encodeProjectFile, projectFileName } from '@/lib/projectFile'
import { speakerDisplayName, toJson, toSrt, toTxt } from '@/lib/transcriptExport'
import { Activity, AudioLines, Check, ChevronDown, CircleHelp, CloudOff, Download, FileAudio2, FileJson2, FileText, FolderOpen, HardDrive, LoaderCircle, Mic2, Pause, Play, Search, Settings2, ShieldCheck, Sparkles, Trash2, Upload, Volume2, X, Save } from 'lucide-react'

type Model = { id: string; name: string; description?: string; loaded?: boolean; cached?: boolean; size?: string; family?: string; publisher?: string; family_publisher?: string; publisher_slug?: string; publisher_url?: string; source_url?: string; license?: string; runtime?: string; capabilities?: { computeTypes?: boolean; beamSize?: boolean; vad?: boolean; initialPrompt?: boolean; languageHint?: string } }
type Segment = { start: number; end: number; text: string; id: string; speaker?: string }
type Job = { id: string; status: string; progress?: number; segments?: unknown[]; error?: string; [key: string]: unknown }
type ProjectSettings = { model: string; compute: string; language: string; beam: number; vad: boolean; prompt: string }
type Project = { id: string; schemaVersion: 1; name: string; audio: Blob; fileName: string; mimeType: string; settings: ProjectSettings; segments: Segment[]; createdAt: number; savedAt: number; duration?: number; speakerNames?: Record<string, string> }
const PROJECT_DB = 'stt-studio-local-projects'
const PROJECT_STORE = 'projects'
function openProjectDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(PROJECT_DB, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(PROJECT_STORE, { keyPath: 'id' })
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error || new Error('Could not open local project storage.'))
  })
}
async function readProjects(): Promise<Project[]> {
  const db = await openProjectDB()
  return new Promise((resolve, reject) => { const tx = db.transaction(PROJECT_STORE, 'readonly'), request = tx.objectStore(PROJECT_STORE).getAll(); request.onsuccess = () => resolve((request.result as Project[]).sort((a,b)=>b.savedAt-a.savedAt)); request.onerror = () => reject(request.error || new Error('Could not read saved projects.')); tx.oncomplete = () => db.close(); tx.onerror = () => { db.close(); reject(tx.error) } })
}
async function writeProject(project: Project): Promise<void> {
  const db = await openProjectDB()
  return new Promise((resolve, reject) => { const tx = db.transaction(PROJECT_STORE, 'readwrite'); tx.objectStore(PROJECT_STORE).put(project); tx.oncomplete = () => { db.close(); resolve() }; tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not save this project.')) }; tx.onabort = () => { db.close(); reject(tx.error || new Error('Could not save this project.')) } })
}
async function removeProject(id: string): Promise<void> {
  const db = await openProjectDB()
  return new Promise((resolve, reject) => { const tx = db.transaction(PROJECT_STORE, 'readwrite'); tx.objectStore(PROJECT_STORE).delete(id); tx.oncomplete = () => { db.close(); resolve() }; tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not delete this project.')) } })
}
const presets: Model[] = [
  { id: 'large-v3', name: 'Large v3', description: 'Best accuracy · 3 GB', size: '3 GB' },
  { id: 'medium', name: 'Medium', description: 'Balanced · 1.5 GB', size: '1.5 GB' },
  { id: 'small', name: 'Small', description: 'Quick · 500 MB', size: '500 MB' },
  { id: 'base', name: 'Base', description: 'Lightweight · 150 MB', size: '150 MB' },
]
const cx = (...xs: Array<string | false | undefined | null>) => xs.filter(Boolean).join(' ')
function timecode(sec: number) { const s = Math.max(0, Math.floor(sec || 0)); return `${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` }
function fmtBytes(n: number) { if (n < 1024 ** 2) return `${Math.round(n / 1024)} KB`; return `${(n / 1024 ** 2).toFixed(1)} MB` }
function normalizeSegments(raw: unknown): Segment[] {
  if (!Array.isArray(raw)) return []
  return raw.map((item: any, i) => ({ id: String(i), start: Number(item.start ?? item.start_time ?? 0), end: Number(item.end ?? item.end_time ?? 0), text: String(item.text ?? ''), ...(item.speaker ? { speaker: String(item.speaker) } : {}) }))
}
function cleanSpeakerNames(names: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(names).map(([label, name]) => [label, name.trim()]).filter(([, name]) => name))
}
async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) { let detail = ''; try { const data = await response.json(); detail = data.detail || data.error || data.message || '' } catch { detail = await response.text() }; throw Object.assign(new Error(detail || `Request failed (${response.status})`), { status: response.status }) }
  return response.json()
}
function modelList(payload: any): Model[] {
  const rows = Array.isArray(payload) ? payload : payload.models ?? payload.available_models ?? payload.items ?? []
  return rows.map((m: any) => typeof m === 'string' ? { id: m, name: m } : ({ id: m.id ?? m.name ?? m.model, name: m.display_name ?? m.label ?? m.name ?? m.id ?? m.model, description: m.description ?? m.size, loaded: Boolean(m.loaded ?? m.is_loaded), cached: Boolean(m.cached), size: m.size, family: m.family, publisher: m.publisher, family_publisher: m.family_publisher, publisher_slug: m.publisher_slug, publisher_url: m.publisher_url, source_url: m.source_url, license: m.license, runtime: m.runtime, capabilities: m.capabilities }))
}
function ModelOptions({ models }: { models: Model[] }) {
  const families = [...new Set(models.map(item => item.family || 'Whisper'))]
  return <>{families.map(family => <optgroup key={family} label={family}>{models.filter(item => (item.family || 'Whisper') === family).map(item => <option key={item.id} value={item.id}>{item.name}{item.size ? ` · ${item.size}` : ''}{item.cached ? ' · downloaded' : ''}</option>)}</optgroup>)}</>
}
function PublisherMark({ model }: { model?: Model }) {
  const [image, setImage] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    setImage('')
    if (!model?.publisher) return () => controller.abort()
    const org = (model.publisher_slug || model.publisher.toLowerCase()).toLowerCase()
    fetch(`https://huggingface.co/api/organizations/${encodeURIComponent(org)}/avatar`, { signal: controller.signal })
      .then(response => response.ok ? response.json() : fetch(`https://huggingface.co/api/users/${encodeURIComponent(org)}/avatar`, { signal: controller.signal }).then(user => user.ok ? user.json() : null))
      .then(data => { if (typeof data?.avatarUrl === 'string' && /^https:\/\//.test(data.avatarUrl)) setImage(data.avatarUrl) })
      .catch(() => {})
    return () => controller.abort()
  }, [model?.publisher, model?.publisher_slug])
  return <a className="model-symbol" href={model?.publisher_url} target="_blank" rel="noreferrer" aria-label={`Open ${model?.publisher || 'publisher'} profile on Hugging Face`} title={`${model?.publisher || 'Model'} · profile image from Hugging Face`}>
    {image ? <img src={image} alt="" referrerPolicy="no-referrer" onError={() => setImage('')}/> : <Sparkles size={16}/>}
  </a>
}
const LOST_JOB_MESSAGE = 'The local engine no longer has this job. It may have restarted, which clears jobs. Transcribe the file again.'
function prettyError(error: unknown) {
  const message = error instanceof Error ? error.message : 'Something went wrong.'
  if (/failed to fetch|networkerror/i.test(message)) return 'Can’t reach the local transcription service. Check that the STT backend is running, then retry.'
  if (/ffmpeg/i.test(message)) return 'Audio decoder unavailable. Install ffmpeg and restart the local service.'
  return message
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null), importRef = useRef<HTMLInputElement>(null), audioRef = useRef<HTMLAudioElement>(null)
  const [health, setHealth] = useState<'checking'|'online'|'offline'>('checking'), [models, setModels] = useState<Model[]>(presets.map(m=>({...m}))), [modelsError, setModelsError] = useState(''), [computeTypes, setComputeTypes] = useState(['int8','float16','float32']), [loadedModels, setLoadedModels] = useState<Array<{model:string;compute_type:string;runtime?:string}>>([])
  const [customModelId, setCustomModelId] = useState('')
  const [speakerSupport, setSpeakerSupport] = useState<SpeakerSupport>(UNKNOWN_SPEAKER_SUPPORT), [labelSpeakers, setLabelSpeakers] = useState(false), [speakerNames, setSpeakerNames] = useState<Record<string, string>>({})
  const [file, setFile] = useState<File | null>(null), [dragging, setDragging] = useState(false), [model, setModel] = useState('small'), [compute, setCompute] = useState('int8'), [language, setLanguage] = useState('auto'), [beam, setBeam] = useState(5), [vad, setVad] = useState(true), [prompt, setPrompt] = useState('')
  const [job, setJob] = useState<Job | null>(null), [segments, setSegments] = useState<Segment[]>([]), [busy, setBusy] = useState(false), [modelLoading, setModelLoading] = useState(false), [pollingPaused, setPollingPaused] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('')
  const [audioUrl, setAudioUrl] = useState(''), [playing, setPlaying] = useState(false), [currentTime, setCurrentTime] = useState(0), [query, setQuery] = useState(''), [exportOpen, setExportOpen] = useState(false), [modelFilter, setModelFilter] = useState('')
  const [projectId, setProjectId] = useState(''), [projectName, setProjectName] = useState(''), [projectDirty, setProjectDirty] = useState(false), [projectList, setProjectList] = useState<Project[]>([])
  const [saveOpen, setSaveOpen] = useState(false), [libraryOpen, setLibraryOpen] = useState(false), [projectListLoading, setProjectListLoading] = useState(false), [projectNameInput, setProjectNameInput] = useState(''), [pendingOpen, setPendingOpen] = useState<Project | null>(null), [deleteTarget, setDeleteTarget] = useState<Project | null>(null), [projectError, setProjectError] = useState('')
  const [pendingFile, setPendingFile] = useState<File | null>(null), [removeConfirm, setRemoveConfirm] = useState(false), [rerunConfirm, setRerunConfirm] = useState(false), [savingProject, setSavingProject] = useState(false)
  const savingRef = useRef(false)
  const done = Boolean(job && /completed|complete|done|succeeded|success/i.test(job.status))
  const running = Boolean(job && !done && !/failed|error|cancel/i.test(job.status))
  const partialRows = normalizeSegments(job?.segments)
  const diarizationReport = (job?.info as { diarization?: { status?: string; reason?: string } } | null | undefined)?.diarization
  const rowsOnScreen = running ? partialRows : segments
  const speakerLabels = [...new Set(rowsOnScreen.flatMap(s => s.speaker ? [s.speaker] : []))]
  const speakerTag = (speaker?: string) => speaker ? <span className={cx('speaker-tag', `speaker-tone-${speakerLabels.indexOf(speaker) % 4}`)} title={speaker}>{speakerDisplayName(speaker, speakerNames)}</span> : null
  const isProcessing = busy || running || modelLoading
  const selectedRuntime = models.find(item=>item.id===model)?.runtime || 'faster-whisper'
  const visibleModels = useMemo(() => { const term = modelFilter.trim().toLowerCase(); if (!term) return models; return models.filter(item => item.id === model || [item.name, item.family, item.publisher, item.runtime, item.id].some(value => String(value ?? '').toLowerCase().includes(term))) }, [models, modelFilter, model])
  // Whisper defaults apply until the API says otherwise, so older responses keep working.
  const caps = { computeTypes: true, beamSize: true, vad: true, initialPrompt: true, languageHint: 'codes', ...models.find(item=>item.id===model)?.capabilities }
  const effectiveCompute = caps.computeTypes ? compute : 'native'
  const modelIsLoaded = loadedModels.some(item=>item.model===model&&item.compute_type===effectiveCompute&&(item.runtime || 'faster-whisper')===selectedRuntime)
  const rawProgress = job?.progress ?? job?.percent ?? job?.progress_percent
  const parsedProgress = rawProgress == null ? null : Number(rawProgress)
  const progress = parsedProgress == null || !Number.isFinite(parsedProgress) ? null : Math.max(0, Math.min(100, parsedProgress))
  const statusLine = job?.cancel_requested ? 'Cancelling…' : job?.status === 'queued' ? 'Waiting in queue' : job?.diarize && progress === 100 ? 'Labelling speakers' : partialRows.length ? `Transcribing, ${partialRows.length} ${partialRows.length === 1 ? 'segment' : 'segments'} so far` : 'Transcribing audio'

  const refresh = useCallback(async () => {
    try { const data: any = await api('/api/health'); setHealth(data.status === 'ok' ? 'online' : 'offline') } catch { setHealth('offline') }
    try {
      const data: any = await api('/api/models')
      const loaded: Array<{model:string;compute_type:string;runtime?:string}> = Array.isArray(data.loaded_models) ? data.loaded_models : []
      const available = modelList(data)
      setLoadedModels(loaded)
      if (!available.length) available.push(...presets.map(m=>({...m})))
      setModels(available); if (Array.isArray(data.compute_types) && data.compute_types.length) setComputeTypes(data.compute_types)
      setModelsError('')
    } catch (e) { setModelsError(prettyError(e)) }
    try {
      const data: any = await api('/api/capabilities')
      setSpeakerSupport({ checked: true, available: Boolean(data.diarization?.available), reason: data.diarization?.reason || (data.diarization?.available ? null : 'Speaker labels are not available on this server.') })
    } catch { setSpeakerSupport({ checked: true, available: false, reason: 'Could not read speaker label support from the local engine.' }) }
  }, [])
  useEffect(() => { refresh(); const id = window.setInterval(refresh, 15000); return () => window.clearInterval(id) }, [refresh])
  useEffect(() => { if (audioUrl) return () => URL.revokeObjectURL(audioUrl) }, [audioUrl])
  useEffect(() => {
    if (!job?.id || !running || pollingPaused) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const next: any = await api(`/api/jobs/${encodeURIComponent(job.id)}`)
        if (cancelled) return
        const updated = { ...next, id: String(next.id ?? job.id) } as Job
        setJob(updated)
        if (updated.status === 'failed') { setBusy(false); setError(prettyError(new Error(String(updated.error || 'Transcription failed.')))) }
        if (updated.status === 'cancelled') { setBusy(false); const kept = normalizeSegments(updated.segments); setSegments(kept); setNotice(kept.length ? `Transcription cancelled. Kept ${kept.length} segment${kept.length === 1 ? '' : 's'} transcribed so far.` : 'Transcription cancelled before any speech was transcribed.'); window.setTimeout(() => setNotice(''), 6000) }
        if (/^(queued|running)$/.test(updated.status)) timer = window.setTimeout(poll, 1000)
      } catch (e) {
        if (cancelled) return
        if ((e as { status?: number }).status === 404) { setBusy(false); setJob(prev => prev && { ...prev, status: 'failed', error: LOST_JOB_MESSAGE }); setError(LOST_JOB_MESSAGE); return }
        setBusy(false); setPollingPaused(true); setError(prettyError(e))
      }
    }
    void poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [job?.id, running, pollingPaused])
  useEffect(() => { const a = audioRef.current; if (!a) return; const tick = () => setCurrentTime(a.currentTime); const end = () => setPlaying(false); a.addEventListener('timeupdate', tick); a.addEventListener('ended', end); return () => { a.removeEventListener('timeupdate', tick); a.removeEventListener('ended', end) } }, [audioUrl])
  useEffect(() => { if (done && job) { const rows = normalizeSegments(job.segments ?? (job.result as any)?.segments ?? (job.transcription as any)?.segments); if (rows.length) setSegments(rows); setBusy(false) } }, [done, job])
  const applyFile = (f: File) => { setError(''); setJob(null); setSegments([]); setSpeakerNames({}); setFile(f); setPlaying(false); setProjectId(''); setProjectName(''); setProjectDirty(true); if (audioUrl) URL.revokeObjectURL(audioUrl); setAudioUrl(URL.createObjectURL(f)); setPendingFile(null) }
  const selectFile = (f?: File) => { if (!f || isProcessing) return; if (!f.type.startsWith('audio/') && !/\.(wav|mp3|m4a|aac|flac|ogg|opus|wma|mp4|webm)$/i.test(f.name)) { setError('Choose an audio file such as WAV, MP3, M4A, FLAC, or OGG.'); return }; if (file && projectDirty) { setPendingFile(f); return }; applyFile(f) }
  const removeCurrentFile = () => { setFile(null); setJob(null); setSegments([]); setSpeakerNames({}); setProjectId(''); setProjectName(''); setProjectDirty(false); setAudioUrl(''); setCurrentTime(0); setRemoveConfirm(false) }
  const requestRemoveFile = () => { if (projectDirty && file) setRemoveConfirm(true); else removeCurrentFile() }
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDragging(false); if (!isProcessing) selectFile(e.dataTransfer.files[0]) }
  const loadModel = async () => { setModelLoading(true); setError(''); try { await api('/api/models/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, compute_type: effectiveCompute, runtime: models.find(item=>item.id===model)?.runtime || 'faster-whisper' }) }); await refresh(); setNotice('Model is ready on this Mac.'); window.setTimeout(() => setNotice(''), 4000) } catch(e) { setError(prettyError(e)) } finally { setModelLoading(false) } }
  const addCustomModel = () => {
    const id = customModelId.trim()
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(id)) { setError('Enter a Hugging Face owner/repository ID, such as Systran/faster-whisper-large-v3.'); return }
    setModels(current => current.some(item => item.id === id) ? current : [...current, { id, name: id, family: 'custom CTranslate2', runtime: 'faster-whisper', publisher: id.split('/')[0], publisher_slug: id.split('/')[0], publisher_url: `https://huggingface.co/${id.split('/')[0]}`, source_url: `https://huggingface.co/${id}` }])
    setModel(id); setCustomModelId(''); setProjectDirty(true); setError('')
  }
  const performTranscribe = async () => {
    if (!file) { inputRef.current?.click(); return }
    if (health === 'offline') { setError('The local transcription service is offline. Start the backend and try again.'); return }
    setBusy(true); setError(''); setNotice(''); setSegments([]); setSpeakerNames({}); setJob(null); setProjectDirty(true)
    const sentLanguage = caps.languageHint === 'auto' ? 'auto' : language
    const body = new FormData(); body.append('file', file); body.append('model', model); body.append('runtime', models.find(item=>item.id===model)?.runtime || 'faster-whisper'); body.append('compute_type', effectiveCompute); if (sentLanguage !== 'auto') body.append('language', sentLanguage); body.append('beam_size', String(beam)); body.append('vad_filter', String(vad)); body.append('initial_prompt', prompt); body.append('diarize', String(speakerSupport.available && labelSpeakers))
    try { const result: any = await api('/api/jobs', { method: 'POST', body }); const normalized = { ...result, id: String(result.id ?? result.job_id ?? result.job?.id ?? ''), status: String(result.status ?? result.job?.status ?? 'queued') } as Job; if (!normalized.id) throw new Error('The server started the job but did not return an ID.'); setPollingPaused(false); setJob(normalized); const rows = normalizeSegments(result.segments ?? result.result?.segments); if (rows.length) setSegments(rows) } catch(e) { setError(prettyError(e)); setBusy(false) }
  }
  const cancelJob = async () => { if (!job?.id) return; try { const next: any = await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' }); setJob(prev => ({ ...(prev ?? {}), ...next, id: String(next.id ?? job.id) } as Job)) } catch (e) { setError(prettyError(e)) } }
  const transcribe = () => { if (isProcessing) return; if (!file) { inputRef.current?.click(); return }; if (segments.length) { setRerunConfirm(true); return }; void performTranscribe() }
  const filtered = useMemo(() => query.trim() ? segments.map((s,i)=>({s,i})).filter(({s})=>s.text.toLowerCase().includes(query.toLowerCase())).map(({s})=>s) : segments, [segments, query])
  const playAt = (time: number) => { if (!audioRef.current) return; audioRef.current.currentTime = time; audioRef.current.play().then(()=>setPlaying(true)).catch(()=>{}) }
  const togglePlay = () => { if (!audioRef.current) return; if (audioRef.current.paused) audioRef.current.play().then(()=>setPlaying(true)).catch(()=>{}); else { audioRef.current.pause(); setPlaying(false) } }
  const updateSegment = (id: string, text: string) => { setSegments(prev => prev.map(s=>s.id===id?{...s,text}:s)); setProjectDirty(true) }
  const renameSpeaker = (label: string, name: string) => { setSpeakerNames(prev => ({ ...prev, [label]: name })); setProjectDirty(true) }
  const exportFile = (type: 'txt'|'srt'|'json') => {
    const content = type === 'srt' ? toSrt(segments, speakerNames) : type === 'json' ? toJson(segments, speakerNames) : toTxt(segments, speakerNames)
    const mime = type === 'srt' ? 'application/x-subrip' : type === 'json' ? 'application/json' : 'text/plain'
    const url = URL.createObjectURL(new Blob([content],{type:mime})), a=document.createElement('a'); a.href=url; a.download=`${file?.name.replace(/\.[^.]+$/,'') || 'transcript'}.${type}`; a.click(); URL.revokeObjectURL(url); setExportOpen(false)
  }
  const copy = async () => { try { await navigator.clipboard.writeText(toTxt(segments, speakerNames)); setNotice('Transcript copied to clipboard.'); window.setTimeout(()=>setNotice(''),3000) } catch { setError('Clipboard access was blocked by the browser.') } }
  const refreshProjects = async () => { setProjectListLoading(true); setProjectError(''); try { setProjectList(await readProjects()) } catch (e) { setProjectError(prettyError(e)) } finally { setProjectListLoading(false) } }
  const beginSave = () => {
    if (!file) { setError('Choose an audio file before saving a project.'); return }
    if (running || busy || modelLoading) { setError('Wait for the current operation to finish before saving this project.'); return }
    setError(''); setProjectError(''); setProjectNameInput(projectName || file.name.replace(/\.[^.]+$/, '')); setSaveOpen(true)
  }
  const commitSave = async () => {
    if (!file) return
    const name = projectNameInput.trim()
    if (!name) { setProjectError('Enter a name for this project.'); return }
    if (savingRef.current) return
    savingRef.current = true; setSavingProject(true)
    try {
      const previous = projectId ? projectList.find(p=>p.id===projectId) : undefined
      const project: Project = { id: projectId || (crypto.randomUUID?.() ?? `project-${Date.now()}-${Math.random().toString(36).slice(2)}`), schemaVersion: 1, name, audio: file.slice(0,file.size,file.type || 'application/octet-stream'), fileName: file.name, mimeType: file.type || 'application/octet-stream', settings: {model,compute,language,beam,vad,prompt}, segments: segments.map(({id,start,end,text,speaker})=>({id,start,end,text,...(speaker?{speaker}:{})})), speakerNames: cleanSpeakerNames(speakerNames), createdAt: previous?.createdAt ?? Date.now(), savedAt: Date.now(), duration: audioRef.current?.duration || undefined }
      await writeProject(project); setProjectId(project.id); setProjectName(name); setProjectDirty(false); setSaveOpen(false); setNotice('Project saved in this browser on this Mac.'); window.setTimeout(()=>setNotice(''),4000); await refreshProjects()
    } catch (e) { const message = e instanceof DOMException && e.name === 'QuotaExceededError' ? 'Browser storage is full. Delete an older project or free space, then try again.' : prettyError(e); setProjectError(message) } finally { savingRef.current = false; setSavingProject(false) }
  }
  const openLibrary = async () => { if (isProcessing) { setError('Wait for the current operation to finish before opening a project.'); return }; setLibraryOpen(true); setError(''); await refreshProjects() }
  const activateProject = (project: Project) => { if (isProcessing) { setError('Wait for the current operation to finish before opening a project.'); return }; setLibraryOpen(false); if (projectDirty && file) setPendingOpen(project); else void restoreProject(project) }
  const restoreProject = async (project: Project) => {
    try {
      if (project.schemaVersion !== 1 || !(project.audio instanceof Blob) || !project.settings || !Array.isArray(project.segments)) throw new Error('This saved project uses an unsupported or incomplete format.')
      const restoredFile = new File([project.audio], project.fileName, {type:project.mimeType})
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(URL.createObjectURL(project.audio)); setFile(restoredFile); setPlaying(false); setCurrentTime(0); setSegments(project.segments.map((s,i)=>({...s,id:String(i)}))); setSpeakerNames(project.speakerNames ?? {}); setJob(null); setBusy(false); setPollingPaused(false)
      if (!models.some(item=>item.id===project.settings.model) && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(project.settings.model)) setModels(current=>[...current,{id:project.settings.model,name:project.settings.model,family:'custom CTranslate2',runtime:'faster-whisper',publisher:project.settings.model.split('/')[0],publisher_slug:project.settings.model.split('/')[0],publisher_url:`https://huggingface.co/${project.settings.model.split('/')[0]}`,source_url:`https://huggingface.co/${project.settings.model}`}])
      setModel(project.settings.model); setCompute(project.settings.compute); setLanguage(project.settings.language); setBeam(project.settings.beam); setVad(project.settings.vad); setPrompt(project.settings.prompt)
      setProjectId(project.id); setProjectName(project.name); setProjectDirty(false); setPendingOpen(null); setLibraryOpen(false); setError(''); setNotice(project.id ? `Opened “${project.name}” from local browser storage.` : `Imported “${project.name}” from a project file. Save it to keep a copy in this browser.`); window.setTimeout(()=>setNotice(''),4000)
    } catch (e) { setError(prettyError(e)); setPendingOpen(null) }
  }
  const exportProjectFile = () => {
    if (!file) { setError('Choose an audio file before exporting a project.'); return }
    if (running || busy || modelLoading) { setError('Wait for the current operation to finish before exporting this project.'); return }
    try {
      const name = projectName.trim() || file.name.replace(/\.[^.]+$/, '')
      const saved = projectId ? projectList.find(p=>p.id===projectId) : undefined
      const blob = encodeProjectFile({ name, createdAt: saved?.createdAt ?? Date.now(), savedAt: Date.now(), duration: audioRef.current?.duration || undefined,
        settings: {model,compute,language,beam,vad,prompt}, segments: segments.map(({start,end,text,speaker})=>({start,end,text,...(speaker?{speaker}:{})})), speakerNames: cleanSpeakerNames(speakerNames),
        audio: file, fileName: file.name, mimeType: file.type || 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a'); link.href = url; link.download = projectFileName(name); document.body.appendChild(link); link.click(); link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setNotice(`Exported “${name}” as a project file.`); window.setTimeout(()=>setNotice(''),4000)
    } catch (e) { setError(e instanceof ProjectFileError ? e.message : prettyError(e)) }
  }
  const importProjectFile = async (picked: File) => {
    if (isProcessing) { setError('Wait for the current operation to finish before importing a project.'); return }
    setError(''); setProjectError('')
    try {
      const { manifest, audio } = await decodeProjectFile(picked)
      // An empty id marks an imported copy that is not in browser storage until the user saves it.
      const project: Project = { id: '', schemaVersion: 1, name: manifest.name, audio, fileName: manifest.audio.fileName, mimeType: manifest.audio.mimeType,
        settings: manifest.settings, segments: manifest.segments.map((s,i)=>({id:String(i), ...s})), speakerNames: manifest.speakerNames ?? {}, createdAt: manifest.createdAt, savedAt: manifest.savedAt, duration: manifest.duration }
      if (projectDirty && file) setPendingOpen(project); else await restoreProject(project)
    } catch (e) { setError(e instanceof ProjectFileError ? e.message : prettyError(e)) }
  }
  const confirmDeleteProject = async () => {
    if (!deleteTarget) return
    try { await removeProject(deleteTarget.id); if (projectId===deleteTarget.id) { setProjectId(''); setProjectName(''); setProjectDirty(Boolean(file)) }; setDeleteTarget(null); await refreshProjects(); setNotice('Saved project deleted from this browser.'); window.setTimeout(()=>setNotice(''),4000) } catch (e) { setProjectError(prettyError(e)); setDeleteTarget(null) }
  }
  const handleKey = (e: KeyboardEvent<HTMLDivElement>) => { const target=e.target as HTMLElement; if(target.closest('dialog[open]')||target.matches('input,textarea,select,[contenteditable="true"]')||target.isContentEditable)return; if (e.key === 'Escape' && exportOpen) setExportOpen(false); if (e.key==='Enter' && (e.metaKey||e.ctrlKey) && (!isProcessing||pollingPaused)) { e.preventDefault(); if (pollingPaused) { setError(''); setPollingPaused(false); setBusy(true) } else transcribe() } }
  return <div className="app-shell" onKeyDown={handleKey} onClick={e=>{if (!(e.target as HTMLElement).closest('.export-wrap')) setExportOpen(false)}}>
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><AudioLines size={21}/></div><div><strong>STT Studio</strong><span>LOCAL TRANSCRIPTION</span></div></div>
      <div className="nav-item active" aria-current="page"><AudioLines size={17}/><span>Workspace</span><span className="nav-dot"/></div>
      <div className="side-rule"/><div className="side-label">LOCAL PROCESSING</div>
      <div className="sidebar-bottom"><div className="privacy-card"><div className="privacy-icon"><ShieldCheck size={17}/></div><div><strong>Private by design</strong><p>Audio stays on your Mac.<br/>No cloud upload.</p></div></div><div className="service-row"><span className={cx('status-led',health==='online'&&'is-online',health==='checking'&&'is-checking')}/><span>{health==='online'?'Local engine ready':health==='checking'?'Connecting to engine…':'Engine offline'}</span><button className="icon-button micro" onClick={refresh} aria-label="Reconnect to local engine"><Activity size={14}/></button></div><div className="version">STT STUDIO <span>·</span> 0.1.0</div></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>{projectName || (file ? file.name : 'New transcription')}{projectDirty&&projectId&&<i className="dirty-mark" title="Unsaved project changes"> •</i>}</strong></div><div className="top-actions"><button className="top-action" onClick={openLibrary} disabled={isProcessing||savingProject} aria-label="Open project"><FolderOpen size={15}/><span>Open project</span></button><button className="top-action" onClick={()=>importRef.current?.click()} disabled={isProcessing||savingProject} aria-label="Import project file"><Upload size={15}/><span>Import file</span></button><button className="top-action" onClick={exportProjectFile} disabled={!file||isProcessing||savingProject} aria-label="Download project file"><Download size={15}/><span>Download file</span></button><button className="top-action save-action" onClick={beginSave} disabled={!file||isProcessing||savingProject} aria-label={projectId?'Save changes':'Save project'}><Save size={15}/><span>{projectId?'Save changes':'Save project'}</span></button><span className={cx('connection-pill',health==='online'?'connected':health==='offline'?'disconnected':'')}><span className="status-led"/>{health==='online'?'Local engine online':health==='offline'?'Engine offline':'Checking connection'}</span></div></header>
      <div className="workspace-content"><div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line"/> AUDIO TO TEXT</div><h1>Transcription workspace</h1><p className="page-subtitle">Accurate speech recognition, entirely on your device.</p></div><div className="local-chip"><HardDrive size={14}/> Runs locally <span className="chip-sep">·</span> Private</div></div>
        <div className="work-grid">
          <section className="left-column">
            <div className="panel source-panel"><div className="panel-heading"><div className="step-icon">01</div><div><h2>Choose an audio file</h2><p>Drop it in, or browse your Mac.</p></div><button className="help-button" aria-label="Supported audio formats" title="WAV, MP3, M4A, AAC, FLAC, OGG, OPUS"><CircleHelp size={16}/></button></div>
              <div className={cx('dropzone',dragging&&'dragging',file&&'has-file',isProcessing&&'locked')} aria-disabled={isProcessing} onDrop={onDrop} onDragOver={e=>{e.preventDefault();if(!isProcessing)setDragging(true)}} onDragLeave={()=>setDragging(false)}>
                <input ref={inputRef} type="file" accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.wma,.mp4,.webm" onChange={(e:ChangeEvent<HTMLInputElement>)=>{selectFile(e.target.files?.[0]);e.target.value=''}} hidden disabled={isProcessing}/>
                {file ? <><div className="file-type-icon"><FileAudio2 size={23}/></div><div className="file-copy"><strong title={file.name}>{file.name}</strong><span>{fmtBytes(file.size)} <i/> {isProcessing?'Transcription in progress':'Ready to transcribe'}</span></div><button className="icon-button" disabled={isProcessing} onClick={requestRemoveFile} aria-label="Remove selected audio"><X size={17}/></button></> : <><div className="upload-icon"><Upload size={20}/></div><div className="drop-copy"><strong>Drop audio file here</strong><span>or <button className="text-button" onClick={()=>inputRef.current?.click()} disabled={isProcessing}>browse files</button></span></div><div className="formats">WAV <i/> MP3 <i/> M4A <i/> FLAC</div></>}
              </div>
              {audioUrl && <div className="audio-player"><button className="play-button" onClick={togglePlay} aria-label={playing?'Pause':'Play'}>{playing?<Pause size={16} fill="currentColor"/>:<Play size={16} fill="currentColor"/>}</button><div className="audio-track"><div className="audio-track-top"><span>{file?.name}</span><span>{timecode(currentTime)} <i>/</i> {timecode(audioRef.current?.duration || 0)}</span></div><input aria-label="Audio position" className="seekbar" type="range" min="0" max={audioRef.current?.duration || 100} value={currentTime} onChange={e=>{if(audioRef.current)audioRef.current.currentTime=Number(e.target.value)}} style={{'--seek':`${audioRef.current?.duration?currentTime/audioRef.current.duration*100:0}%`} as React.CSSProperties}/></div><Volume2 size={15} className="muted-icon"/><audio ref={audioRef} src={audioUrl}/></div>}
            </div>
            <div className="panel config-panel"><div className="panel-heading"><div className="step-icon">02</div><div><h2>Configure your run</h2><p>Choose a model and fine-tune recognition.</p></div></div>
              <div className="model-filter"><input type="search" aria-label="Search list" placeholder="Filter models by name, family, publisher or runtime" value={modelFilter} onChange={e=>setModelFilter(e.target.value)} disabled={isProcessing}/>{modelFilter.trim()&&<span className="field-hint">{visibleModels.length} of {models.length} shown</span>}</div>
              <div className="config-grid"><label className="field"><span className="field-label">MODEL</span><span className="select-wrap"><select value={model} disabled={isProcessing} onChange={e=>{setModel(e.target.value);setProjectDirty(true)}}><ModelOptions models={visibleModels}/></select><ChevronDown size={15}/></span><span className="field-hint">{models.find(m=>m.id===model)?.description || 'Higher accuracy may take longer.'}</span></label>
                <label className="field"><span className="field-label">LANGUAGE</span><span className="select-wrap"><select value={caps.languageHint==='auto'?'auto':language} disabled={isProcessing||caps.languageHint==='auto'} onChange={e=>{setLanguage(e.target.value);setProjectDirty(true)}}><option value="auto">Auto detect</option><option value="en">English</option><option value="vi">Vietnamese</option><option value="es">Spanish</option><option value="fr">French</option><option value="de">German</option><option value="ja">Japanese</option><option value="zh">Chinese</option><option value="ko">Korean</option></select><ChevronDown size={15}/></span><span className="field-hint">{caps.languageHint==='auto'?'This model detects the language on its own.':'Detected automatically when enabled.'}</span></label>
              </div>
              <SpeakerLabelOption checked={labelSpeakers} support={speakerSupport} disabled={isProcessing} onChange={setLabelSpeakers}/><details className="advanced"><summary><span><Settings2 size={14}/> Advanced options</span><ChevronDown size={14}/></summary><div className="advanced-body">{caps.computeTypes&&<label className="field"><span className="field-label">COMPUTE TYPE</span><span className="select-wrap"><select value={compute} disabled={isProcessing} onChange={e=>{setCompute(e.target.value);setProjectDirty(true)}}>{computeTypes.map(type=><option key={type} value={type}>{type === 'int8' ? 'Int8 · less memory' : type === 'float16' ? 'Float16 · faster on GPU' : type}</option>)}</select><ChevronDown size={15}/></span></label>}{caps.beamSize&&<label className="field"><span className="field-label">BEAM SIZE <b>{beam}</b></span><input type="range" min="1" max="10" value={beam} disabled={isProcessing} onChange={e=>{setBeam(Number(e.target.value));setProjectDirty(true)}}/><span className="field-hint">Larger beams can improve accuracy, with slower processing.</span></label>}{caps.vad&&<label className="toggle-row"><span><strong>Voice activity detection</strong><small>Skip silent sections for faster results.</small></span><Switch.Root checked={vad} disabled={isProcessing} onCheckedChange={value=>{setVad(value);setProjectDirty(true)}} className="radix-switch" aria-label="Voice activity detection"><Switch.Thumb className="radix-switch-thumb"/></Switch.Root></label>}{caps.initialPrompt&&<label className="field prompt-field"><span className="field-label">VOCABULARY HINT <em>OPTIONAL</em></span><textarea value={prompt} disabled={isProcessing} onChange={e=>{setPrompt(e.target.value);setProjectDirty(true)}} placeholder="Names, technical terms, or phrases to recognize…" rows={2}/></label>}<div className="field custom-model-field"><span className="field-label">CUSTOM MODEL REPOSITORY</span><div className="custom-model-input"><input value={customModelId} disabled={isProcessing} onChange={e=>setCustomModelId(e.target.value)} placeholder="owner/model (CTranslate2 Whisper)"/><button type="button" className="button-secondary" disabled={isProcessing||!customModelId.trim()} onClick={addCustomModel}>Add</button></div><span className="field-hint">Downloads on demand. The repository must contain a CTranslate2 Whisper model.</span></div></div></details>
              <div className="model-row"><div className="model-meta"><PublisherMark model={models.find(m=>m.id===model)}/><span><strong>{models.find(m=>m.id===model)?.name || model}</strong><small>{modelIsLoaded?'Model loaded and ready':models.find(m=>m.id===model)?.cached?'Downloaded · load to warm':`Not loaded for ${effectiveCompute}`}</small></span></div><Button variant="secondary" onClick={loadModel} disabled={isProcessing||health==='offline'}>{modelLoading?<><LoaderCircle size={15} className="spin"/> Loading…</>:<><Download size={15}/> Load and warm model</>}</Button></div>
              {models.find(m=>m.id===model)&&<div className="model-provenance"><span>{models.find(m=>m.id===model)?.family || 'custom model'}</span><span>{models.find(m=>m.id===model)?.runtime || 'CTranslate2'}</span>{models.find(m=>m.id===model)?.family_publisher&&models.find(m=>m.id===model)?.family_publisher!==models.find(m=>m.id===model)?.publisher&&<span>Upstream: {models.find(m=>m.id===model)?.family_publisher}</span>}{models.find(m=>m.id===model)?.license&&<span>{models.find(m=>m.id===model)?.license} license</span>}<a href={models.find(m=>m.id===model)?.source_url || models.find(m=>m.id===model)?.publisher_url} target="_blank" rel="noreferrer">Model card ↗</a><small>Publisher profile image from Hugging Face · size is approximate</small></div>}
            </div>
            <div className="run-row"><div className="run-note"><ShieldCheck size={15}/><span>Processing happens entirely on this device</span></div><button className="button-primary" onClick={()=>pollingPaused?(setError(''),setPollingPaused(false),setBusy(true)):transcribe()} disabled={(isProcessing&&!pollingPaused)||(!file&&!pollingPaused)}><span>{pollingPaused?<Activity size={16}/>:busy?<LoaderCircle size={16} className="spin"/>:<Mic2 size={16}/>}</span>{pollingPaused?'Retry status check':busy?'Transcribing…':'Transcribe audio'}{!busy&&!pollingPaused&&<span className="shortcut">⌘ ↵</span>}</button></div>
            {error&&<div className="alert error-alert" role="alert"><div><strong>Something needs attention</strong><span>{error}</span></div><button className="icon-button" onClick={()=>setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
            {notice&&<div className="alert success-alert" role="status"><Check size={16}/>{notice}</div>}
            {modelsError&&health==='online'&&<p className="subtle-error">Model list unavailable. Showing common models.</p>}
          </section>
          <section className="panel transcript-panel"><div className="transcript-head"><div><div className="eyebrow small-eyebrow"><span className="eyebrow-line"/> OUTPUT</div><div className="transcript-title"><h2>Transcript</h2>{segments.length>0&&<span className="segment-count">{segments.length} segments</span>}</div></div><div className="output-actions"><button className="icon-button" disabled={!segments.length} onClick={copy} title="Copy transcript" aria-label="Copy transcript"><FileText size={16}/></button><div className="export-wrap"><button className="export-button" disabled={!segments.length} aria-haspopup="menu" aria-expanded={exportOpen} onClick={()=>setExportOpen(value=>!value)}><Download size={15}/> Export <ChevronDown size={13}/></button>{exportOpen&&<div className="export-menu open" role="menu"><button role="menuitem" onClick={()=>exportFile('txt')}><FileText size={14}/> Plain text (.txt)</button><button role="menuitem" onClick={()=>exportFile('srt')}><SubtitlesIcon/> SubRip captions (.srt)</button><button role="menuitem" onClick={()=>exportFile('json')}><FileJson2 size={14}/> Segment data (.json)</button></div>}</div></div></div>
              {!running&&diarizationReport?.status==='failed'&&<div className="speaker-failure"><strong>Speaker labels could not be produced.</strong><span>{diarizationReport.reason || 'No reason was given.'} The transcript is still usable.</span></div>}
              {running&&<div className="progress-card"><div className="progress-top"><div className="progress-status"><LoaderCircle size={15} className="spin"/><span>{statusLine}</span></div>{progress!==null&&<strong>{progress}%</strong>}</div>{progress!==null?<div className="progress-bar"><span style={{width:`${progress}%`}}/></div>:<div className="progress-bar indeterminate"><span/></div>}<div className="progress-bottom"><span>{file?.name}</span><span>{progress===null?'Progress details unavailable':'Keep this window open'}</span></div><div className="progress-actions"><button className="button-secondary" onClick={cancelJob} disabled={Boolean(job?.cancel_requested)}><X size={13}/> {job?.cancel_requested?'Cancelling…':'Cancel transcription'}</button></div></div>}
              {running&&partialRows.length>0&&<div className="partial-transcript" aria-live="polite"><div className="partial-head">Live partial transcript · read-only until the job finishes, then speaker names can be added</div>{partialRows.map((s,i)=><div className="partial-row" key={i}><span className="partial-time">{timecode(s.start)}</span><p>{speakerTag(s.speaker)}{s.text}</p></div>)}</div>}
              {!segments.length ? <div className="transcript-empty">{running ? <><div className="empty-orb running-orb"><AudioLines size={24}/></div><h3>Your words are on their way</h3><p>The transcript will appear here when processing is complete.</p><div className="waveform" aria-hidden="true">{Array.from({length:34},(_,i)=><i key={i} style={{height:`${8+Math.round(Math.abs(Math.sin(i*.8)*30 + Math.cos(i*.31)*12))}px`,animationDelay:`${i*35}ms`}}/>)}</div></> : done ? <><div className="empty-orb"><AudioLines size={23}/></div><h3>No speech segments found</h3><p>The job completed without returning transcript text.</p></> : job&&/failed|error/i.test(job.status)?<><div className="empty-orb warning-orb"><X size={21}/></div><h3>Transcription didn’t finish</h3><p>{error || job.error || 'Check the local engine and retry.'}</p></> : <><div className="empty-orb"><AudioLines size={23}/></div><h3>Your transcript will appear here</h3><p>Add an audio file and start a transcription.<br/>You can edit every segment once it’s ready.</p><div className="empty-hint"><span>01</span> Select audio <i/> <span>02</span> Transcribe <i/> <span>03</span> Edit & export</div></>}</div> : <><div className="transcript-toolbar"><div className="search-box"><Search size={14}/><input aria-label="Search transcript" placeholder="Find in transcript…" value={query} onChange={e=>setQuery(e.target.value)}/>{query&&<button onClick={()=>setQuery('')} aria-label="Clear search"><X size={13}/></button>}</div><span className="edit-label"><span/> Click any line to edit</span></div>{speakerLabels.length>0&&<div className="speaker-names"><span className="field-hint">Names replace the speaker labels on screen and in exports. Clear a name to show the original label again.</span><div className="speaker-name-grid">{speakerLabels.map(label=><div className="speaker-name-field" key={label}><span>{label}</span><input aria-label={`Name for ${label}`} value={speakerNames[label]??''} placeholder={label} maxLength={MAX_SPEAKER_NAME} onChange={e=>renameSpeaker(label,e.target.value)}/></div>)}</div></div>}<div className="segments-list">{filtered.map((s)=><div className={cx('segment-row',currentTime>=s.start&&currentTime<s.end&&playing&&'segment-active')} key={s.id}><button className="timestamp" onClick={()=>playAt(s.start)} title="Play from this timestamp">{timecode(s.start)}</button><div className="segment-text">{speakerTag(s.speaker)}<textarea aria-label={`Transcript segment at ${timecode(s.start)}`} value={s.text} onChange={e=>updateSegment(s.id,e.target.value)} rows={Math.max(1,Math.ceil(s.text.length/72))}/></div></div>)}{!filtered.length&&<div className="no-results">No matching lines. Try another search.</div>}</div><div className="transcript-foot"><span><Check size={13}/> Edits apply to exports</span><span><CloudOff size={14}/>{projectId?(projectDirty?'Unsaved edits · saved copy unchanged':'Saved in this browser'): 'Session only until you save'}</span></div></>}
          </section>
        </div>
        <BatchPanel settings={{ model, runtime: models.find(item=>item.id===model)?.runtime || 'faster-whisper', compute: effectiveCompute, language: caps.languageHint === 'auto' ? 'auto' : language, beam, vad, prompt }} online={health==='online'} disabled={isProcessing} speakerSupport={speakerSupport} labelSpeakers={labelSpeakers} onLabelSpeakersChange={setLabelSpeakers}/>
        <footer className="app-footer"><span><ShieldCheck size={13}/> Your audio never leaves this Mac</span><span>Built for focused listening <i>✳</i></span></footer>
      </div>
    </main>
    <input ref={importRef} type="file" accept={PROJECT_FILE_EXTENSION} hidden aria-label="Project file to import" onChange={(e:ChangeEvent<HTMLInputElement>)=>{const picked=e.target.files?.[0]; e.target.value=''; if (picked) void importProjectFile(picked)}}/>
    <Modal open={saveOpen} title={projectId?'Save project changes':'Save a local project'} onClose={()=>{if(!savingProject)setSaveOpen(false)}}>
      <form onSubmit={e=>{e.preventDefault();void commitSave()}}>
        <p className="dialog-copy">The audio file, transcript edits, and transcription settings will be saved in this browser on this Mac. Nothing is uploaded. Saving is always manual. Browser cleanup or clearing site data may remove saved projects. Export transcript files separately for a text or caption backup; transcript exports do not include audio or settings.</p>
        <label className="dialog-label" htmlFor="project-name">Project name</label><input id="project-name" className="dialog-input" data-autofocus maxLength={100} value={projectNameInput} onChange={e=>setProjectNameInput(e.target.value)} placeholder="Give this project a name"/>
        {projectId&&<p className="dialog-note">This replaces the saved version of “{projectName}” with the current workspace contents.</p>}
        {projectError&&<div className="dialog-error" role="alert">{projectError}</div>}
        <div className="dialog-actions"><button type="button" className="button-secondary" onClick={()=>setSaveOpen(false)} disabled={savingProject}>Cancel</button><button type="submit" className="button-primary" disabled={savingProject}><Save size={15}/>{savingProject?'Saving…':projectId?'Save changes':'Save project'}</button></div>
      </form>
    </Modal>
    <Modal open={libraryOpen} title="Local projects" onClose={()=>setLibraryOpen(false)} wide>
      <p className="dialog-copy">Projects are stored only in this browser for this local app address on this Mac. Browser cleanup or clearing site data may remove them. Opening restores audio, transcript edits, and settings; backend job IDs are never saved. Export transcript files separately for a text or caption backup; transcript exports do not include audio or settings.</p>
      {projectError&&<div className="dialog-error" role="alert">{projectError}</div>}{projectListLoading?<div className="project-loading"><LoaderCircle size={18} className="spin"/> Reading local projects…</div>:!projectList.length?<div className="project-library-empty"><div className="empty-orb"><FolderOpen size={22}/></div><h3>No saved projects yet</h3><p>Choose <b>Save project</b> after selecting audio to keep a local copy you can reopen later.</p></div>:<div className="project-list">{projectList.map(project=><article className="project-card" key={project.id}><div className="project-card-icon"><FileAudio2 size={18}/></div><div className="project-card-info"><strong>{project.name}</strong><span title={project.fileName}>{project.fileName}</span><small>{new Date(project.savedAt).toLocaleString()} <i/> {project.segments.length} transcript segments</small></div><button className="button-secondary project-open" onClick={()=>activateProject(project)} disabled={running||busy||modelLoading}><FolderOpen size={14}/> Open</button><button className="icon-button delete-project" onClick={()=>{setProjectError('');setDeleteTarget(project)}} aria-label={`Delete saved project ${project.name}`} title="Delete project"><Trash2 size={16}/></button></article>)}</div>}
      <div className="dialog-actions library-actions"><span className="storage-location"><HardDrive size={14}/> Browser storage on this Mac</span><button type="button" className="button-secondary" onClick={()=>setLibraryOpen(false)}>Close</button></div>
    </Modal>
    <Modal open={Boolean(pendingOpen)} title="Replace the current workspace?" onClose={()=>setPendingOpen(null)}>
      <p className="dialog-copy">There are unsaved changes in the current workspace. Opening “{pendingOpen?.name}” will replace them. Choose Keep working if you want to save the current work first.</p>
      <div className="dialog-actions"><button className="button-secondary" onClick={()=>setPendingOpen(null)}>Keep working</button><button className="button-danger" onClick={()=>pendingOpen&&void restoreProject(pendingOpen)}>Discard and open</button></div>
    </Modal>
    <Modal open={Boolean(pendingFile)} title="Replace the current workspace?" onClose={()=>setPendingFile(null)}>
      <p className="dialog-copy">Unsaved transcript edits and settings will be discarded when you replace this audio with “{pendingFile?.name}”. Cancel and save a project first if you want to keep them.</p>
      <div className="dialog-actions"><button className="button-secondary" onClick={()=>setPendingFile(null)}>Keep current audio</button><button className="button-danger" onClick={()=>pendingFile&&applyFile(pendingFile)}>Discard changes and replace audio</button></div>
    </Modal>
    <Modal open={removeConfirm} title="Remove this audio?" onClose={()=>setRemoveConfirm(false)}>
      <p className="dialog-copy">This removes the current audio and discards unsaved transcript edits and settings from the workspace. Any saved project remains in the local library.</p>
      <div className="dialog-actions"><button className="button-secondary" onClick={()=>setRemoveConfirm(false)}>Keep working</button><button className="button-danger" onClick={removeCurrentFile}>Discard changes and remove audio</button></div>
    </Modal>
    <Modal open={rerunConfirm} title="Run transcription again?" onClose={()=>setRerunConfirm(false)}>
      <p className="dialog-copy">A new run replaces the current transcript and any edits in this workspace. The last saved project version stays unchanged. Save the current work first if you want to keep it.</p>
      <div className="dialog-actions"><button className="button-secondary" onClick={()=>setRerunConfirm(false)}>Keep current transcript</button><button className="button-danger" onClick={()=>{setRerunConfirm(false);void performTranscribe()}}>Discard edits and run again</button></div>
    </Modal>
    <Modal open={Boolean(deleteTarget)} title="Delete saved project?" onClose={()=>setDeleteTarget(null)}>
      <p className="dialog-copy">“{deleteTarget?.name}” and its saved audio, transcript, and settings will be removed from this browser. This cannot be undone. If this project is open, the current workspace copy will stay open.</p>
      <div className="dialog-actions"><button className="button-secondary" onClick={()=>setDeleteTarget(null)}>Cancel</button><button className="button-danger" onClick={()=>void confirmDeleteProject()}><Trash2 size={14}/> Delete project</button></div>
    </Modal>
  </div>
}
function SubtitlesIcon() { return <span className="srt-icon">SRT</span> }
function Modal({open,title,onClose,children,wide=false}:{open:boolean;title:string;onClose:()=>void;children:React.ReactNode;wide?:boolean}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(()=>{const dialog=ref.current;if(!dialog)return;if(open&&!dialog.open){dialog.showModal();requestAnimationFrame(()=>dialog.querySelector<HTMLElement>('[data-autofocus]')?.focus())}else if(!open&&dialog.open)dialog.close()},[open])
  return <dialog ref={ref} className={cx('app-dialog',wide&&'wide-dialog')} aria-labelledby={`dialog-${title.replace(/[^a-z0-9]+/gi,'-')}`} onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===ref.current)onClose()}}><div className="dialog-heading"><h2 id={`dialog-${title.replace(/[^a-z0-9]+/gi,'-')}`}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={17}/></button></div><div className="dialog-body">{children}</div></dialog>
}
