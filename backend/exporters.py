"""Transcript exporters shared by the HTTP API and the agent CLI.

SRT times use HH:MM:SS,mmm and VTT uses HH:MM:SS.mmm. Milliseconds are rounded the same way
as the browser export so a file downloaded in the UI matches one produced by the API.
"""
from __future__ import annotations

import json
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


def _clock(seconds: float, separator: str) -> str:
    total_ms = round(max(0.0, float(seconds or 0)) * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _clean_rows(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in segments:
        text = str(row.get("text", "")).strip()
        if text:
            rows.append({"start": float(row.get("start", 0) or 0), "end": float(row.get("end", 0) or 0), "text": text})
    return rows


def to_txt(segments: Iterable[dict[str, Any]]) -> str:
    return "\n".join(row["text"] for row in _clean_rows(segments)) + ("\n" if segments else "")


def to_srt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(_clean_rows(segments), start=1):
        blocks.append(f"{index}\n{_clock(row['start'], ',')} --> {_clock(row['end'], ',')}\n{row['text']}\n")
    return "\n".join(blocks)


def to_vtt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for row in _clean_rows(segments):
        blocks.append(f"{_clock(row['start'], '.')} --> {_clock(row['end'], '.')}\n{row['text']}\n")
    return "\n".join(blocks)


def to_json(segments: Iterable[dict[str, Any]]) -> str:
    rows = [{"start": row["start"], "end": row["end"], "text": row["text"]} for row in _clean_rows(segments)]
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
        header = f"== {item.get('filename') or 'audio'} ({item.get('status')}) =="
        body = to_txt(item.get("segments") or []).rstrip("\n")
        if item.get("error"):
            body = f"Error: {item['error']}"
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
             "segments": [{"start": r["start"], "end": r["end"], "text": r["text"]} for r in _clean_rows(item.get("segments") or [])]}
            for item in files
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
