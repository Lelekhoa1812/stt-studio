"""Verify model-card metadata and downloadable artifacts against the Hugging Face API.

Pass --offline to check only the catalog's local capability metadata. Offline runs make no network requests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

CATALOG = Path(__file__).resolve().parents[1] / "models" / "catalog.json"
API = "https://huggingface.co/api/models/{}"
CONTROL_KEYS = ("language", "vad", "beamSize", "initialPrompt", "wordTimestamps", "diarization", "hotwords",
                "translation")
EMIT_KEYS = ("segments", "wordTimestamps", "detectedLanguage")


def fetch_model(repository: str) -> dict:
    url = API.format(quote(repository, safe="/"))
    request = Request(url, headers={"User-Agent": "STT-Studio-model-catalog-check/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _check_flags(label: str, name: str, flags: object, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(flags, dict):
        return [f"{label}: {name} object is missing"]
    issues = [f"{label}: {name}.{key} is missing" for key in keys if key not in flags]
    issues += [f"{label}: {name}.{key} must be true or false" for key in keys
               if key in flags and not isinstance(flags[key], bool)]
    issues += [f"{label}: {name}.{key} is not a known key" for key in flags if key not in keys]
    return issues


def check_capability_metadata(path: Path = CATALOG) -> list[str]:
    """Offline: every model entry declares its controls, emits, cancellation and download size."""
    catalog = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    for family in catalog["families"]:
        for model in family["models"]:
            label = f"{family['name']} / {model.get('label', model.get('id'))} ({model.get('repository')})"
            issues.extend(_check_flags(label, "controls", model.get("controls"), CONTROL_KEYS))
            issues.extend(_check_flags(label, "emits", model.get("emits"), EMIT_KEYS))
            if not isinstance(model.get("cancellation"), bool):
                issues.append(f"{label}: cancellation must be true or false")
            size = model.get("downloadSize", "missing")
            if size == "missing" or not (size is None or isinstance(size, str)):
                issues.append(f"{label}: downloadSize must be a string or null")
    return issues


def check_catalog(path: Path = CATALOG) -> list[str]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    for family in catalog["families"]:
        for model in family["models"]:
            repository = model["repository"]
            label = f"{family['name']} / {model['label']} ({repository})"
            try:
                metadata = fetch_model(repository)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                issues.append(f"{label}: model card lookup failed: {exc}")
                continue
            tags = {tag.lower() for tag in metadata.get("tags", [])}
            if metadata.get("private") or metadata.get("gated") or metadata.get("disabled"):
                issues.append(f"{label}: repository is private, gated, or disabled")
            license_tag = f"license:{model['license'].lower()}"
            if license_tag not in tags:
                issues.append(f"{label}: expected {license_tag}, found {[tag for tag in tags if tag.startswith('license:')]}")
            files = {entry.get("rfilename") for entry in metadata.get("siblings", [])}
            if family.get("runtimeStatus") == "supported":
                runtime = family.get("runtimeId", "faster-whisper")
                if runtime == "faster-whisper" and "model.bin" not in files:
                    issues.append(f"{label}: no model.bin found; repository may not be CTranslate2-compatible")
                if runtime == "mlx-audio" and not any(name and name.endswith(".safetensors") for name in files):
                    issues.append(f"{label}: no .safetensors weights found; repository may not be an MLX conversion")
            print(f"{'OK' if not any(issue.startswith(label) for issue in issues) else 'CHECK'}  {label}")
            time.sleep(0.1)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the model catalog against Hugging Face model cards.")
    parser.add_argument("--offline", action="store_true",
                        help="check only the local capability metadata; make no network requests")
    args = parser.parse_args(argv)
    try:
        problems = check_capability_metadata()
        if not args.offline:
            problems += check_catalog()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Catalog verification failed: {exc}", file=sys.stderr)
        return 2
    if problems:
        print("\nCatalog issues:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("\nOffline catalog metadata checks passed." if args.offline else "\nModel catalog checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
