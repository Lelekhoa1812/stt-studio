// Client-side transcript exports. They mirror backend/exporters.py byte for byte when no speaker is renamed.

export type TranscriptRow = { start: number; end: number; text: string; speaker?: string }
export type SpeakerNames = Record<string, string>

export function speakerDisplayName(speaker: string, names: SpeakerNames = {}): string {
  return names[speaker]?.trim() || speaker
}

type CleanRow = { start: number; end: number; text: string; speaker?: string }

// Python's str.isspace() set, which str.strip() and \s use on the backend. JavaScript's \s and trim() differ
// (they include U+FEFF and omit U+001C to U+001F and U+0085), so the set is spelled out.
const PY_WHITESPACE = '\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
const PY_STRIP = new RegExp(`^[${PY_WHITESPACE}]+|[${PY_WHITESPACE}]+$`, 'g')
// Mirrors backend/exporters.py _LINE_BREAK: a run containing a line break, with its surrounding whitespace, becomes one space.
const LINE_BREAK = new RegExp(`[${PY_WHITESPACE}]*[\\n\\v\\f\\r\\x1c-\\x1e\\x85\\u2028\\u2029]+[${PY_WHITESPACE}]*`, 'g')

function pyStrip(value: string): string {
  return value.replace(PY_STRIP, '')
}

// Keeps a cue on one line, and keeps '-->' out of it so the timing line is the only one that has it.
function cueText(value: string): string {
  let text = value.replace(LINE_BREAK, ' ')
  while (text.includes('-->')) text = text.split('-->').join('->')
  return text
}

function vttEscape(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function cleanRows(segments: TranscriptRow[], names: SpeakerNames): CleanRow[] {
  return segments.flatMap(row => {
    const text = pyStrip(String(row.text ?? ''))
    if (!text) return []
    const start = Number(row.start) || 0
    return [{
      start,
      end: Math.max(start, Number(row.end) || 0),
      text,
      ...(row.speaker ? { speaker: speakerDisplayName(row.speaker, names) } : {}),
    }]
  })
}

// Python's round() uses half-to-even, so exact half milliseconds must round the same way.
function roundHalfEven(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (fraction < 0.5) return floor
  if (fraction > 0.5) return floor + 1
  return floor % 2 === 0 ? floor : floor + 1
}

const pad = (value: number, width: number) => String(value).padStart(width, '0')

function clock(seconds: number, separator: ',' | '.'): string {
  const total = roundHalfEven(Math.max(0, Number(seconds) || 0) * 1000)
  const hours = Math.floor(total / 3_600_000)
  const minutes = Math.floor(total / 60_000) % 60
  const secs = Math.floor(total / 1000) % 60
  const millis = total % 1000
  return `${pad(hours, 2)}:${pad(minutes, 2)}:${pad(secs, 2)}${separator}${pad(millis, 3)}`
}

// Python writes floats as 2.0 and 1e-05, which JSON.stringify would render as 2 and 0.00001.
function pythonFloat(value: number): string {
  if (Number.isFinite(value) && value !== 0 && (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e16)) {
    const [mantissa, exponent] = value.toExponential().split('e')
    const sign = exponent.startsWith('-') ? '-' : '+'
    return `${mantissa}e${sign}${exponent.replace(/^[+-]/, '').padStart(2, '0')}`
  }
  const text = String(value)
  return /[.e]/.test(text) ? text : `${text}.0`
}

export function toTxt(segments: TranscriptRow[], names: SpeakerNames = {}): string {
  const rows = cleanRows(segments, names)
  if (!rows.length) return ''
  return rows.map(row => (row.speaker ? `${row.speaker}: ${row.text}` : row.text)).join('\n') + '\n'
}

export function toSrt(segments: TranscriptRow[], names: SpeakerNames = {}): string {
  return cleanRows(segments, names).map((row, index) => {
    const text = cueText(row.speaker ? `${row.speaker}: ${row.text}` : row.text)
    return `${index + 1}\n${clock(row.start, ',')} --> ${clock(row.end, ',')}\n${text}\n`
  }).join('\n')
}

export function toVtt(segments: TranscriptRow[], names: SpeakerNames = {}): string {
  const blocks = cleanRows(segments, names).map(row => {
    const text = row.speaker
      ? `<v ${vttEscape(cueText(row.speaker))}>${vttEscape(cueText(row.text))}</v>`
      : vttEscape(cueText(row.text))
    return `${clock(row.start, '.')} --> ${clock(row.end, '.')}\n${text}\n`
  })
  return ['WEBVTT\n', ...blocks].join('\n')
}

export function toJson(segments: TranscriptRow[], names: SpeakerNames = {}): string {
  const rows = cleanRows(segments, names)
  if (!rows.length) return '[]\n'
  const objects = rows.map(row => {
    const fields = [`"start": ${pythonFloat(row.start)}`, `"end": ${pythonFloat(row.end)}`, `"text": ${JSON.stringify(row.text)}`]
    if (row.speaker) fields.push(`"speaker": ${JSON.stringify(row.speaker)}`)
    return `  {\n${fields.map(field => `    ${field}`).join(',\n')}\n  }`
  })
  return `[\n${objects.join(',\n')}\n]\n`
}
