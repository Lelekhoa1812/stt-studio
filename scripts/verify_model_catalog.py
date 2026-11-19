"""Verify model-card metadata and downloadable artifacts against the Hugging Face API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

CATALOG = Path(__file__).resolve().parents[1] / "models" / "catalog.json"
API = "https://huggingface.co/api/models/{}"


def fetch_model(repository: str) -> dict:
    url = API.format(quote(repository, safe="/"))
    request = Request(url, headers={"User-Agent": "STT-Studio-model-catalog-check/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


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
            if family.get("runtimeStatus") == "supported" and "model.bin" not in files:
                issues.append(f"{label}: no model.bin found; repository may not be CTranslate2-compatible")
            print(f"{'OK' if not any(issue.startswith(label) for issue in issues) else 'CHECK'}  {label}")
            time.sleep(0.1)
    return issues


if __name__ == "__main__":
    try:
        problems = check_catalog()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Catalog verification failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if problems:
        print("\nCatalog issues:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("\nModel catalog checks passed.")
