"""Transcript exporters shared by the HTTP API and the agent CLI.

SRT times use HH:MM:SS,mmm and VTT uses HH:MM:SS.mmm. Milliseconds are rounded the same way
as the browser export so a file downloaded in the UI matches one produced by the API.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

EXPORT_FORMATS = ("txt", "srt", "vtt", "json")
BATCH_EXPORT_FORMATS = ("txt", "json")

MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
}

BATCH_FORMAT_NAME = "stt-studio-batch-results"


_LINE_BREAK = re.compile(r"\s*[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]+\s*")


def _cue_text(value: str) -> str:
    """Keep a cue on one line, and keep '-->' out of it so the timing line is the only one that has it."""
    text = _LINE_BREAK.sub(" ", value)
    while "-->" in text:
        text = text.replace("-->", "->")
    return text


def _vtt_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clock(seconds: float, separator: str) -> str:
    total_ms = round(max(0.0, float(seconds or 0)) * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _one_line(value: Any) -> str:
    """Replace line breaks so a header or an error stays on one line."""
    return _LINE_BREAK.sub(" ", str(value)).strip()


def _finite_time(value: Any) -> float | None:
    """A usable cue time, or None when the value is missing a finite number."""
    if value is None or value == "":
        value = 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean_rows(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in segments:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        start = _finite_time(row.get("start", 0))
        end = _finite_time(row.get("end", 0))
        if start is None or end is None:
            continue
        cleaned = {"start": start, "end": max(start, end), "text": text}
        if row.get("speaker"):
            cleaned["speaker"] = str(row["speaker"])
        rows.append(cleaned)
    return rows


def to_txt(segments: Iterable[dict[str, Any]]) -> str:
    lines = [f"{row['speaker']}: {row['text']}" if "speaker" in row else row["text"] for row in _clean_rows(segments)]
    return "\n".join(lines) + "\n" if lines else ""


def to_srt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(_clean_rows(segments), start=1):
        text = f"{row['speaker']}: {row['text']}" if "speaker" in row else row["text"]
        text = _cue_text(text)
        blocks.append(f"{index}\n{_clock(row['start'], ',')} --> {_clock(row['end'], ',')}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for row in _clean_rows(segments):
        if "speaker" in row:
            text = f"<v {_vtt_escape(_cue_text(row['speaker']))}>{_vtt_escape(_cue_text(row['text']))}</v>"
        else:
            text = _vtt_escape(_cue_text(row["text"]))
        blocks.append(f"{_clock(row['start'], '.')} --> {_clock(row['end'], '.')}\n{text}\n")
    return "\n".join(blocks)


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {"start": row["start"], "end": row["end"], "text": row["text"]}
    if "speaker" in row:
        out["speaker"] = row["speaker"]
    return out


def to_json(segments: Iterable[dict[str, Any]]) -> str:
    rows = [_json_row(row) for row in _clean_rows(segments)]
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


_RENDERERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def render(segments: Iterable[dict[str, Any]], fmt: str) -> str:
    """Render one transcript. Raises ValueError for an unknown format."""
    if fmt not in _RENDERERS:
        raise ValueError(f"format must be one of: {', '.join(EXPORT_FORMATS)}")
    return _RENDERERS[fmt](list(segments))


def batch_text(files: list[dict[str, Any]]) -> str:
    """Plain text for a batch: one headed section per file, in submission order."""
    sections = []
    for item in files:
        name = _one_line(item.get("filename") or "audio") or "audio"
        header = f"== {name} ({_one_line(item.get('status'))}) =="
        body = to_txt(item.get("segments") or []).rstrip("\n")
        if item.get("error"):
            body = f"Error: {_one_line(item['error'])}"
        sections.append(f"{header}\n{body}\n")
    return "\n".join(sections)


def batch_json(batch_id: str, exported_at: str, files: list[dict[str, Any]]) -> str:
    payload = {
        "format": BATCH_FORMAT_NAME,
        "version": 1,
        "batch_id": batch_id,
        "exported_at": exported_at,
        "files": [
            {"filename": item.get("filename"), "status": item.get("status"), "error": item.get("error"),
             "info": item.get("info"),
             "segments": [_json_row(r) for r in _clean_rows(item.get("segments") or [])]}
            for item in files
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
