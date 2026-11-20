import os
import tempfile

import pytest

# The app opens its job store when backend.app is imported, so point it away from the repository first.
_import_data_dir = tempfile.TemporaryDirectory(prefix="stt-studio-test-")
os.environ["WHISPER_DATA_DIR"] = _import_data_dir.name


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    path = tmp_path / "persist-store"
    monkeypatch.setenv("WHISPER_DATA_DIR", str(path))
    return path
