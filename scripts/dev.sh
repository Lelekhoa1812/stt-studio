#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"

missing=()
for command_name in python3.12 node npm ffmpeg ffprobe; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing+=("$command_name")
  fi
done
if ((${#missing[@]})); then
  printf 'Missing required tools: %s\n' "${missing[*]}" >&2
  printf 'On macOS, install them with Homebrew: brew install python@3.12 node ffmpeg\n' >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
  python3.12 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install -r "$BACKEND_DIR/requirements.txt"

if [[ ! -d "$FRONTEND_DIR/node_modules" || ! -f "$FRONTEND_DIR/node_modules/.package-lock.json" || \
      "$FRONTEND_DIR/package-lock.json" -nt "$FRONTEND_DIR/node_modules/.package-lock.json" ]]; then
  (cd "$FRONTEND_DIR" && npm ci)
fi

frontend_pid=""
stop_services() {
  kill "$backend_pid" 2>/dev/null || true
  [[ -z "$frontend_pid" ]] || kill "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
  [[ -z "$frontend_pid" ]] || wait "$frontend_pid" 2>/dev/null || true
}
"$VENV_DIR/bin/uvicorn" backend.app:app --host 127.0.0.1 --port 8765 &
backend_pid=$!
trap stop_services EXIT INT TERM

backend_ready=false
for _attempt in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8765/api/health >/dev/null; then
    backend_ready=true
    break
  fi
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    printf 'The local backend exited during startup.\n' >&2
    exit 1
  fi
  sleep 1
done
if [[ "$backend_ready" != true ]]; then
  printf 'The local backend did not become ready within 30 seconds.\n' >&2
  exit 1
fi

(
  cd "$FRONTEND_DIR"
  npm run dev
) &
frontend_pid=$!

printf '\nSTT Studio is starting. Open the local URL printed by Vite.\n'
printf 'Keep this terminal open; press Ctrl+C to stop both services.\n\n'
wait
