"""Compare two transcripts with normalized word error rate (WER)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Keep apostrophes inside a word, while ignoring punctuation and whitespace.
# Combining marks (U+0300-U+036F) belong to the word they follow; \w alone would split "q" + mark + "uick".
_WORD = re.compile(r"[\w\u0300-\u036f]+(?:['’][\w\u0300-\u036f]+)*", re.UNICODE)
# A comma between digit groups is a thousands separator: "1,000" is the same number as "1000".
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TEXT_KEYS = ("transcript", "text", "segments", "result", "transcription")


@dataclass(frozen=True)
class WERResult:
    reference_words: int
    hypothesis_words: int
    substitutions: int
    insertions: int
    deletions: int
    errors: int
    wer: float | None


def tokenize(text: str) -> list[str]:
    """Normalize Unicode, case and punctuation, then return word tokens."""
    normalized = unicodedata.normalize("NFC", text)
    # Casefolding U+0130 (dotted capital I) adds a combining dot; Turkish and English readers spell it "i".
    normalized = _THOUSANDS_SEPARATOR.sub("", normalized.replace("\u0130", "I")).casefold()
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return _WORD.findall(normalized)


def _extract_text(value: Any) -> str:
    """Extract transcript text from a string or common segment JSON shapes."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = [_extract_text(item) for item in value]
        return " ".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            if key in value:
                return _extract_text(value[key])
        raise ValueError(
            "JSON object has no recognized transcript field; expected one of: "
            + ", ".join(_TEXT_KEYS)
        )
    raise ValueError("Transcript JSON must be a string, object, or array of segments")


def read_transcript(path: str | Path) -> str:
    """Read UTF-8 plain text or JSON transcript content from *path*."""
    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: transcript must be valid UTF-8") from exc
    if source.suffix.casefold() == ".json":
        try:
            return _extract_text(json.loads(content))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    return content


def compare(reference: str, hypothesis: str) -> WERResult:
    """Compute minimum-edit word counts and WER using unit-cost Levenshtein alignment.

    WER is undefined when the reference contains no words. In that case, ``wer``
    is ``None`` while insertion and error counts remain available.
    """
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    # Carry the alignment counts alongside each cost. Only the previous row is
    # needed, avoiding a quadratic-size trace table for longer transcripts.
    # Tuples are (cost, substitutions, insertions, deletions).
    previous = [(j, 0, j, 0) for j in range(len(hyp) + 1)]
    for i, reference_word in enumerate(ref, start=1):
        current = [(i, 0, 0, i)]
        for j, hypothesis_word in enumerate(hyp, start=1):
            diagonal = previous[j - 1]
            if reference_word == hypothesis_word:
                current.append(diagonal)
                continue
            cost, substitutions, insertions, deletions = diagonal
            substitution = (cost + 1, substitutions + 1, insertions, deletions)
            cost, substitutions, insertions, deletions = previous[j]
            deletion = (cost + 1, substitutions, insertions, deletions + 1)
            cost, substitutions, insertions, deletions = current[j - 1]
            insertion = (cost + 1, substitutions, insertions + 1, deletions)
            # Stable tie order matches the original traceback: S, D, I.
            current.append(min((substitution, deletion, insertion), key=lambda item: item[0]))
        previous = current

    _, substitutions, insertions, deletions = previous[-1]
    errors = substitutions + insertions + deletions
    wer = errors / len(ref) if ref else None
    return WERResult(len(ref), len(hyp), substitutions, insertions, deletions, errors, wer)


def _render_text(result: WERResult) -> str:
    rate = "undefined (empty reference)" if result.wer is None else f"{result.wer:.2%}"
    return "\n".join((
        f"WER: {rate}",
        f"Reference words: {result.reference_words}",
        f"Hypothesis words: {result.hypothesis_words}",
        f"Substitutions: {result.substitutions}",
        f"Insertions: {result.insertions}",
        f"Deletions: {result.deletions}",
        f"Total errors: {result.errors}",
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare UTF-8 transcript files with normalized word error rate."
    )
    parser.add_argument("reference", help="Reference transcript (.txt or .json)")
    parser.add_argument("hypothesis", help="Hypothesis transcript (.txt or .json)")
    parser.add_argument("--format", choices=("text", "json"), default="text",
                        help="Output format (default: text)")
    args = parser.parse_args(argv)
    try:
        result = compare(read_transcript(args.reference), read_transcript(args.hypothesis))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(_render_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
