#!/usr/bin/env bash
# Shows whether the watch agent is installed, its settings, and the last 20 log lines. Never calls launchctl.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="${STT_WATCH_AGENT_DIR:-$HOME/Library/LaunchAgents}"
case "$agent_dir" in /*) ;; *) echo "error: STT_WATCH_AGENT_DIR must be an absolute path" >&2; exit 2 ;; esac
plist="$agent_dir/com.sttstudio.watch.plist"

state="$(python3 "$HERE/plist.py" state "$plist")"
case "$state" in
    missing)
        echo "Not installed: no plist at $plist"
        echo "Install with: scripts/watch_service/install.sh WATCH_DIR OUTPUT_DIR"
        exit 0
        ;;
    foreign)
        echo "error: $plist was not written by install.sh; not reading it." >&2
        exit 1
        ;;
esac

python3 "$HERE/plist.py" show "$plist"
log="$(python3 "$HERE/plist.py" log "$plist")"
echo
if [ -f "$log" ]; then
    echo "Last 20 lines of $log:"
    tail -n 20 "$log"
else
    echo "No log yet at $log"
fi
