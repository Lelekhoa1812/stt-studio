"""Job, batch and AI subcommands of the CLI. HTTP is mocked: no request reaches a real server."""
import io
import json
import urllib.error

import pytest

from test_stt_cli import cli


@pytest.fixture()
def api(monkeypatch):
    """Record every request and answer from a route table. Each route maps to (response bytes, status or None)."""
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:8765")
    routes: dict[tuple[str, str], object] = {}
    sent: list[dict] = []

    def fake_call(method, path, *, params=None, json_body=None, timeout=60, **kwargs):
        sent.append({"method": method, "path": path, "params": params, "json": json_body})
        response = routes.get((method, path))
        if response is None:
            raise AssertionError(f"unexpected request {method} {path}")
        if isinstance(response, Exception):
            raise response
        return json.dumps(response).encode("utf-8")

    monkeypatch.setattr(cli, "call", fake_call)
    return routes, sent


JOB = {"id": "11111111-2222-3333-4444-555555555555", "status": "completed", "created_at": 1_700_000_000.0,
       "filename": "meeting.wav"}


def test_jobs_list_passes_paging_and_prints_a_table(api, capsys):
    routes, sent = api
    routes[("GET", "/api/jobs")] = {"jobs": [JOB], "total": 3, "limit": 5, "offset": 1}
    assert cli.main(["jobs", "list", "--limit", "5", "--offset", "1"]) == 0
    out = capsys.readouterr().out
    assert sent[0]["params"] == {"limit": 5, "offset": 1}
    assert JOB["id"] in out and "completed" in out and "meeting.wav" in out
    assert "Showing 2-2 of 3 job(s)." in out
    assert "{" not in out  # human output, not JSON


def test_jobs_list_defaults_to_fifty_from_the_start(api, capsys):
    routes, sent = api
    routes[("GET", "/api/jobs")] = {"jobs": [], "total": 0, "limit": 50, "offset": 0}
    assert cli.main(["jobs", "list"]) == 0
    assert sent[0]["params"] == {"limit": 50, "offset": 0}
    assert "No jobs stored." in capsys.readouterr().out


def test_jobs_list_json_prints_the_server_response(api, capsys):
    routes, _ = api
    payload = {"jobs": [JOB], "total": 1, "limit": 50, "offset": 0}
    routes[("GET", "/api/jobs")] = payload
    assert cli.main(["jobs", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload


@pytest.mark.parametrize("argv", [["jobs", "list", "--limit", "0"], ["jobs", "list", "--limit", "201"],
                                  ["jobs", "list", "--offset", "-1"], ["jobs", "list", "--limit", "many"]])
def test_jobs_list_rejects_out_of_range_values_before_any_request(api, capsys, argv):
    _, sent = api
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert sent == []


def test_jobs_delete_sends_a_delete_and_confirms(api, capsys):
    routes, sent = api
    routes[("DELETE", f"/api/jobs/{JOB['id']}")] = {"id": JOB["id"], "deleted": True}
    assert cli.main(["jobs", "delete", JOB["id"]]) == 0
    assert sent == [{"method": "DELETE", "path": f"/api/jobs/{JOB['id']}", "params": None, "json": None}]
    assert f"Deleted job {JOB['id']}." in capsys.readouterr().out


def test_jobs_delete_json_prints_the_response(api, capsys):
    routes, _ = api
    routes[("DELETE", "/api/jobs/job-1")] = {"id": "job-1", "deleted": True}
    assert cli.main(["jobs", "delete", "job-1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"id": "job-1", "deleted": True}


def test_jobs_delete_keeps_a_slash_in_the_id_out_of_the_route(api, capsys):
    routes, sent = api
    routes[("DELETE", "/api/jobs/a%2Fb")] = {"id": "a/b", "deleted": True}
    assert cli.main(["jobs", "delete", "a/b"]) == 0
    assert sent[0]["path"] == "/api/jobs/a%2Fb"


def test_storage_prints_totals_in_human_units(api, capsys):
    routes, sent = api
    routes[("GET", "/api/storage")] = {"data_dir": "data", "jobs": 4, "batches": 1, "bytes": 2048}
    assert cli.main(["storage"]) == 0
    out = capsys.readouterr().out
    assert sent[0]["path"] == "/api/storage"
    assert "Jobs:        4" in out and "Batches:     1" in out and "Disk used:   2.0 KB" in out
    assert "data" in out


def test_storage_json_prints_the_response(api, capsys):
    routes, _ = api
    payload = {"data_dir": "data", "jobs": 4, "batches": 1, "bytes": 2048}
    routes[("GET", "/api/storage")] = payload
    assert cli.main(["storage", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload


@pytest.mark.parametrize("size, expected", [(0, "0 B"), (900, "900 B"), (1536, "1.5 KB"),
                                            (5 * 1024 ** 2, "5.0 MB"), (3 * 1024 ** 3, "3.0 GB")])
def test_human_size_reads_clearly(size, expected):
    assert cli._human_size(size) == expected


def _http_error(monkeypatch, status: int, detail: str):
    """Let the real call() turn a server error response into a CliError, the way a live server would."""
    body = json.dumps({"detail": detail}).encode("utf-8")

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "error", None, io.BytesIO(body))

    monkeypatch.setattr(cli.urllib.request, "urlopen", refuse)


def test_jobs_delete_http_error_exits_1_with_the_server_message(monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:8765")
    _http_error(monkeypatch, 409, "The job is running; cancel it and wait for it to finish before deleting it")
    assert cli.main(["jobs", "delete", "job-1"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: The job is running") and "Traceback" not in err


def test_storage_http_error_with_json_prints_the_message_as_json(monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:8765")
    _http_error(monkeypatch, 503, "Local job storage is unavailable; check disk space and file permissions")
    assert cli.main(["storage", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"].startswith("Local job storage is unavailable")


def test_jobs_list_http_error_exits_1(monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:8765")
    _http_error(monkeypatch, 503, "Local job storage is unavailable; check disk space and file permissions")
    assert cli.main(["jobs", "list"]) == 1
    assert "Local job storage is unavailable" in capsys.readouterr().err


BATCH = {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "created_at": 1_700_000_000.0, "total": 2, "done": True,
         "counts": {"queued": 0, "running": 0, "completed": 1, "failed": 1, "cancelled": 0}}


def test_batches_list_prints_one_line_per_batch(api, capsys):
    routes, sent = api
    routes[("GET", "/api/batches")] = {"batches": [BATCH]}
    assert cli.main(["batches", "list"]) == 0
    out = capsys.readouterr().out
    assert sent[0]["path"] == "/api/batches"
    assert BATCH["id"] in out and "done" in out and "completed 1" in out and "failed 1" in out
    assert "{" not in out


def test_batches_list_accepts_a_bare_list_and_reports_none(api, capsys):
    routes, _ = api
    routes[("GET", "/api/batches")] = []
    assert cli.main(["batches", "list"]) == 0
    assert "No batches stored." in capsys.readouterr().out


def test_batches_list_json_prints_the_response(api, capsys):
    routes, _ = api
    routes[("GET", "/api/batches")] = {"batches": [BATCH]}
    assert cli.main(["batches", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"batches": [BATCH]}


def test_batches_delete_sends_a_delete_and_reports_the_job_count(api, capsys):
    routes, sent = api
    routes[("DELETE", f"/api/batches/{BATCH['id']}")] = {"id": BATCH["id"], "deleted": True, "jobs_deleted": 2}
    assert cli.main(["batches", "delete", BATCH["id"]]) == 0
    assert sent == [{"method": "DELETE", "path": f"/api/batches/{BATCH['id']}", "params": None, "json": None}]
    assert "its 2 job(s)" in capsys.readouterr().out


def test_batches_delete_json_prints_the_response(api, capsys):
    routes, _ = api
    routes[("DELETE", "/api/batches/b1")] = {"id": "b1", "deleted": True, "jobs_deleted": 1}
    assert cli.main(["batches", "delete", "b1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"id": "b1", "deleted": True, "jobs_deleted": 1}


def test_batches_delete_refusal_exits_1_with_the_server_message(monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:8765")
    _http_error(monkeypatch, 409, "A file in this batch is queued or running; cancel the batch and wait before "
                                  "deleting it")
    assert cli.main(["batches", "delete", "b1"]) == 1
    assert "queued or running" in capsys.readouterr().err
