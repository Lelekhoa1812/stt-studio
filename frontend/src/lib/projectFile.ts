// Portable STT Studio project files (.sttproj).
//
// Layout (all integers big-endian):
//   bytes 0-7    magic "STTPRJ01"
//   bytes 8-11   manifest length N (uint32)
//   bytes 12..   manifest, N bytes of UTF-8 JSON
//   rest         raw audio bytes, described by manifest.audio.bytes
//
// The audio is appended as a Blob slice rather than encoded inside JSON, so large recordings are
// never copied into a string or base64-expanded.

export const PROJECT_FILE_EXTENSION = '.sttproj'
export const PROJECT_FILE_MIME = 'application/x-stt-studio-project'

const MAGIC = 'STTPRJ01'
const HEADER_BYTES = MAGIC.length + 4
const MAX_MANIFEST_BYTES = 1024 * 1024
const MAX_SEGMENTS = 100_000
const MAX_SEGMENT_TEXT = 20_000
const MAX_SPEAKER_LABEL = 64
export const MAX_SPEAKER_NAME = 60
const MAX_SPEAKERS = 100

export type ProjectSettings = { model: string; compute: string; language: string; beam: number; vad: boolean; prompt: string }
export type ProjectSegment = { start: number; end: number; text: string; speaker?: string }

export type ProjectFileManifest = {
  format: 'stt-studio-project'
  version: 1
  name: string
  createdAt: number
  savedAt: number
  exportedAt: number
  duration?: number
  audio: { fileName: string; mimeType: string; bytes: number }
  settings: ProjectSettings
  segments: ProjectSegment[]
  // Optional so version 1 files written before speaker names existed still load.
  speakerNames?: Record<string, string>
}

export class ProjectFileError extends Error {}

export type ProjectFileInput = {
  name: string
  createdAt: number
  savedAt: number
  duration?: number
  settings: ProjectSettings
  segments: ProjectSegment[]
  speakerNames?: Record<string, string>
  audio: Blob
  fileName: string
  mimeType: string
}

export function projectFileName(name: string): string {
  const base = name.replace(/[\\/:*?"<>|\x00-\x1f]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120).trim() || 'project'
  return `${base}${PROJECT_FILE_EXTENSION}`
}

export function encodeProjectFile(input: ProjectFileInput, exportedAt: number = Date.now()): Blob {
  const manifest: ProjectFileManifest = {
    format: 'stt-studio-project',
    version: 1,
    name: input.name,
    createdAt: input.createdAt,
    savedAt: input.savedAt,
    exportedAt,
    ...(input.duration != null && Number.isFinite(input.duration) ? { duration: input.duration } : {}),
    audio: { fileName: input.fileName, mimeType: input.mimeType, bytes: input.audio.size },
    settings: input.settings,
    segments: input.segments.map(({ start, end, text, speaker }) => ({ start, end, text, ...(speaker ? { speaker } : {}) })),
    speakerNames: input.speakerNames ?? {},
  }
  const json = new TextEncoder().encode(JSON.stringify(manifest))
  if (json.byteLength > MAX_MANIFEST_BYTES) throw new ProjectFileError('This project has too much transcript text to export as one file.')
  const header = new Uint8Array(HEADER_BYTES)
  header.set(new TextEncoder().encode(MAGIC), 0)
  new DataView(header.buffer).setUint32(MAGIC.length, json.byteLength, false)
  return new Blob([header, json, input.audio], { type: PROJECT_FILE_MIME })
}

function fail(message: string): never {
  throw new ProjectFileError(message)
}

function expectString(value: unknown, label: string, max: number, min = 0): string {
  if (typeof value !== 'string' || value.length < min || value.length > max) fail(`The project file has an invalid ${label}.`)
  return value
}

function expectNumber(value: unknown, label: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) fail(`The project file has an invalid ${label}.`)
  return value
}

export function validateManifest(value: unknown): ProjectFileManifest {
  if (!value || typeof value !== 'object') fail('The project file is not a valid STT Studio project.')
  const m = value as Record<string, unknown>
  if (m.format !== 'stt-studio-project') fail('The file is not an STT Studio project.')
  if (m.version !== 1) fail(`This project file uses format version ${String(m.version)}, which this app cannot read.`)

  const audio = m.audio as Record<string, unknown> | undefined
  const settings = m.settings as Record<string, unknown> | undefined
  if (!audio || typeof audio !== 'object') fail('The project file is missing its audio description.')
  if (!settings || typeof settings !== 'object') fail('The project file is missing its settings.')
  if (!Array.isArray(m.segments) || m.segments.length > MAX_SEGMENTS) fail('The project file has an invalid transcript.')

  const segments = m.segments.map((item, index) => {
    const row = item as Record<string, unknown>
    if (!row || typeof row !== 'object') fail(`Transcript segment ${index + 1} is invalid.`)
    const start = expectNumber(row.start, 'segment start time', 0, Number.MAX_SAFE_INTEGER)
    const end = expectNumber(row.end, 'segment end time', start, Number.MAX_SAFE_INTEGER)
    const text = expectString(row.text, 'segment text', MAX_SEGMENT_TEXT)
    return { start, end, text, ...(row.speaker !== undefined ? { speaker: expectString(row.speaker, 'speaker label', MAX_SPEAKER_LABEL, 1) } : {}) }
  })
  const speakerNames = m.speakerNames === undefined ? undefined : validateSpeakerNames(m.speakerNames)

  const validated: ProjectFileManifest = {
    format: 'stt-studio-project',
    version: 1,
    name: expectString(m.name, 'project name', 200, 1).trim(),
    createdAt: expectNumber(m.createdAt, 'creation time', 0, Number.MAX_SAFE_INTEGER),
    savedAt: expectNumber(m.savedAt, 'save time', 0, Number.MAX_SAFE_INTEGER),
    exportedAt: expectNumber(m.exportedAt, 'export time', 0, Number.MAX_SAFE_INTEGER),
    ...(m.duration != null ? { duration: expectNumber(m.duration, 'duration', 0, 1e9) } : {}),
    audio: {
      fileName: expectString(audio.fileName, 'audio file name', 255, 1),
      mimeType: expectString(audio.mimeType, 'audio type', 200),
      bytes: expectNumber(audio.bytes, 'audio size', 1, Number.MAX_SAFE_INTEGER),
    },
    settings: {
      model: expectString(settings.model, 'model', 200, 1),
      compute: expectString(settings.compute, 'compute type', 64),
      language: expectString(settings.language, 'language', 32),
      beam: expectNumber(settings.beam, 'beam size', 1, 20),
      vad: typeof settings.vad === 'boolean' ? settings.vad : fail('The project file has an invalid VAD setting.'),
      prompt: expectString(settings.prompt, 'vocabulary hint', 2000),
    },
    segments,
    ...(speakerNames !== undefined ? { speakerNames } : {}),
  }
  return validated
}

function validateSpeakerNames(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('The project file has invalid speaker names.')
  const entries = Object.entries(value as Record<string, unknown>)
  if (entries.length > MAX_SPEAKERS) fail('The project file has too many speaker names.')
  return Object.fromEntries(entries.map(([label, name]) => [
    expectString(label, 'speaker label', MAX_SPEAKER_LABEL, 1),
    expectString(name, 'speaker name', MAX_SPEAKER_NAME, 1),
  ]))
}

export async function decodeProjectFile(file: Blob): Promise<{ manifest: ProjectFileManifest; audio: Blob }> {
  if (file.size < HEADER_BYTES) fail('The project file is too small to be an STT Studio project.')
  const header = new Uint8Array(await file.slice(0, HEADER_BYTES).arrayBuffer())
  const magic = new TextDecoder().decode(header.subarray(0, MAGIC.length))
  if (magic !== MAGIC) fail('The file is not an STT Studio project.')
  const manifestLength = new DataView(header.buffer).getUint32(MAGIC.length, false)
  if (manifestLength === 0 || manifestLength > MAX_MANIFEST_BYTES || HEADER_BYTES + manifestLength > file.size) {
    fail('The project file header is damaged.')
  }
  const json = await file.slice(HEADER_BYTES, HEADER_BYTES + manifestLength).text()
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch {
    fail('The project file header is damaged.')
  }
  const manifest = validateManifest(parsed)
  const audio = file.slice(HEADER_BYTES + manifestLength, file.size, manifest.audio.mimeType || 'application/octet-stream')
  if (audio.size !== manifest.audio.bytes) fail('The project file is incomplete: its audio does not match the recorded size.')
  return { manifest, audio }
}
