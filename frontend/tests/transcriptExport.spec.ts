import { expect, test } from '@playwright/test'
import { speakerDisplayName, toJson, toSrt, toTxt, toVtt, type TranscriptRow } from '../src/lib/transcriptExport'

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
  expect(toTxt([{ start: 0, end: 1, text: ' ' }])).toBe('')
  expect(toSrt([{ start: 0, end: 1, text: ' ' }])).toBe('')
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

// Golden cases from backend/exporters.py: line breaks, '-->', VTT escaping, end before start, half-millisecond
// rounding and whitespace that Python strips but JavaScript's trim() does not.
const edgeSegments: TranscriptRow[] = [
  {
    "start": 0.0,
    "end": 1.25,
    "text": "line one\nline two\r\nthree\u2028four\u000bfive \t tab"
  },
  {
    "start": 1.25,
    "end": 2.5,
    "text": "a --> b ---> c",
    "speaker": "Sp --> 1"
  },
  {
    "start": 2.5,
    "end": 2.5,
    "text": "<b>bold</b> & more",
    "speaker": "Ann<1>&\nB"
  },
  {
    "start": 5.0,
    "end": 3.0,
    "text": "backwards end"
  },
  {
    "start": 3.0625,
    "end": 3.1875,
    "text": "half ms"
  },
  {
    "start": 4.0,
    "end": 4.0005,
    "text": "x"
  },
  {
    "start": 6.0,
    "end": 7.0,
    "text": "\u001fpadded\u001f \u3000wide\u00a0"
  },
  {
    "start": 7.0,
    "end": 8.0,
    "text": "  \n  ",
    "speaker": "Ghost"
  },
  {
    "start": 8.0,
    "end": 9.0,
    "text": "tail a\n\n\nb\u2029c",
    "speaker": "Multi\r\nLine"
  }
]

test('edge cases match the backend exporters', () => {
  expect(toTxt(edgeSegments)).toBe("line one\nline two\r\nthree\u2028four\u000bfive \t tab\nSp --> 1: a --> b ---> c\nAnn<1>&\nB: <b>bold</b> & more\nbackwards end\nhalf ms\nx\npadded\u001f \u3000wide\nMulti\r\nLine: tail a\n\n\nb\u2029c\n")
  expect(toSrt(edgeSegments)).toBe("1\n00:00:00,000 --> 00:00:01,250\nline one line two three four five \t tab\n\n2\n00:00:01,250 --> 00:00:02,500\nSp -> 1: a -> b -> c\n\n3\n00:00:02,500 --> 00:00:02,500\nAnn<1>& B: <b>bold</b> & more\n\n4\n00:00:05,000 --> 00:00:05,000\nbackwards end\n\n5\n00:00:03,062 --> 00:00:03,188\nhalf ms\n\n6\n00:00:04,000 --> 00:00:04,000\nx\n\n7\n00:00:06,000 --> 00:00:07,000\npadded\u001f \u3000wide\n\n8\n00:00:08,000 --> 00:00:09,000\nMulti Line: tail a b c\n")
  expect(toVtt(edgeSegments)).toBe("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nline one line two three four five \t tab\n\n00:00:01.250 --> 00:00:02.500\n<v Sp -&gt; 1>a -&gt; b -&gt; c</v>\n\n00:00:02.500 --> 00:00:02.500\n<v Ann&lt;1&gt;&amp; B>&lt;b&gt;bold&lt;/b&gt; &amp; more</v>\n\n00:00:05.000 --> 00:00:05.000\nbackwards end\n\n00:00:03.062 --> 00:00:03.188\nhalf ms\n\n00:00:04.000 --> 00:00:04.000\nx\n\n00:00:06.000 --> 00:00:07.000\npadded\u001f \u3000wide\n\n00:00:08.000 --> 00:00:09.000\n<v Multi Line>tail a b c</v>\n")
  expect(toJson(edgeSegments)).toBe("[\n  {\n    \"start\": 0.0,\n    \"end\": 1.25,\n    \"text\": \"line one\\nline two\\r\\nthree\u2028four\\u000bfive \\t tab\"\n  },\n  {\n    \"start\": 1.25,\n    \"end\": 2.5,\n    \"text\": \"a --> b ---> c\",\n    \"speaker\": \"Sp --> 1\"\n  },\n  {\n    \"start\": 2.5,\n    \"end\": 2.5,\n    \"text\": \"<b>bold</b> & more\",\n    \"speaker\": \"Ann<1>&\\nB\"\n  },\n  {\n    \"start\": 5.0,\n    \"end\": 5.0,\n    \"text\": \"backwards end\"\n  },\n  {\n    \"start\": 3.0625,\n    \"end\": 3.1875,\n    \"text\": \"half ms\"\n  },\n  {\n    \"start\": 4.0,\n    \"end\": 4.0005,\n    \"text\": \"x\"\n  },\n  {\n    \"start\": 6.0,\n    \"end\": 7.0,\n    \"text\": \"padded\\u001f \u3000wide\"\n  },\n  {\n    \"start\": 8.0,\n    \"end\": 9.0,\n    \"text\": \"tail a\\n\\n\\nb\u2029c\",\n    \"speaker\": \"Multi\\r\\nLine\"\n  }\n]\n")
})

test('empty transcripts give empty TXT and SRT, and a bare WEBVTT header', () => {
  expect(toTxt([{ start: 0, end: 1, text: ' ' }])).toBe("")
  expect(toSrt([{ start: 0, end: 1, text: ' ' }])).toBe("")
  expect(toVtt([])).toBe("WEBVTT\n")
  expect(toJson([])).toBe("[]\n")
})
