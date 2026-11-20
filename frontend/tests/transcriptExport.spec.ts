import { expect, test } from '@playwright/test'
import { speakerDisplayName, toJson, toSrt, toTxt } from '../src/lib/transcriptExport'

// Input and expected outputs were produced by running backend/exporters.py on the same segments.
const segments = [
  { start: 0.25, end: 1.5, text: '  hello world  ', speaker: 'Speaker 1' },
  { start: 0.0625, end: 2, text: 'Second, "quoted" line\tTab', speaker: 'Speaker 2' },
  { start: 3, end: 4.5, text: '   ' },
  { start: 4, end: 5, text: 'Plain café 日本' },
  { start: 0.00001, end: 0.0001, text: 'tiny' },
]

const backendTxt = "Speaker 1: hello world\nSpeaker 2: Second, \"quoted\" line\tTab\nPlain caf\u00e9 \u65e5\u672c\ntiny\n"
const backendSrt = "1\n00:00:00,250 --> 00:00:01,500\nSpeaker 1: hello world\n\n2\n00:00:00,062 --> 00:00:02,000\nSpeaker 2: Second, \"quoted\" line\tTab\n\n3\n00:00:04,000 --> 00:00:05,000\nPlain caf\u00e9 \u65e5\u672c\n\n4\n00:00:00,000 --> 00:00:00,000\ntiny\n"
const backendJson = "[\n  {\n    \"start\": 0.25,\n    \"end\": 1.5,\n    \"text\": \"hello world\",\n    \"speaker\": \"Speaker 1\"\n  },\n  {\n    \"start\": 0.0625,\n    \"end\": 2.0,\n    \"text\": \"Second, \\\"quoted\\\" line\\tTab\",\n    \"speaker\": \"Speaker 2\"\n  },\n  {\n    \"start\": 4.0,\n    \"end\": 5.0,\n    \"text\": \"Plain caf\u00e9 \u65e5\u672c\"\n  },\n  {\n    \"start\": 1e-05,\n    \"end\": 0.0001,\n    \"text\": \"tiny\"\n  }\n]\n"

test('unrenamed TXT, SRT and JSON match the backend exporters exactly', () => {
  expect(toTxt(segments)).toBe(backendTxt)
  expect(toSrt(segments)).toBe(backendSrt)
  expect(toJson(segments)).toBe(backendJson)
})

test('empty transcripts match the backend exporters', () => {
  expect(toTxt([])).toBe('')
  expect(toSrt([])).toBe('')
  expect(toJson([])).toBe('[]\n')
  expect(toTxt([{ start: 0, end: 1, text: ' ' }])).toBe('\n')
})

test('renamed speakers replace the anonymous label in every format', () => {
  const names = { 'Speaker 1': 'Alice' }
  expect(toTxt(segments, names)).toBe(backendTxt.replace('Speaker 1: ', 'Alice: '))
  expect(toSrt(segments, names)).toBe(backendSrt.replace('Speaker 1: ', 'Alice: '))
  expect(toJson(segments, names)).toBe(backendJson.replace('"speaker": "Speaker 1"', '"speaker": "Alice"'))
})

test('blank names keep the anonymous label', () => {
  expect(speakerDisplayName('Speaker 2', { 'Speaker 2': '   ' })).toBe('Speaker 2')
  expect(toTxt(segments, { 'Speaker 2': '' })).toBe(backendTxt)
})

test('rounds exact half milliseconds to even, as Python does', () => {
  expect(toSrt([{ start: 0.0625, end: 0.1875, text: 'x' }])).toBe('1\n00:00:00,062 --> 00:00:00,188\nx\n')
})
