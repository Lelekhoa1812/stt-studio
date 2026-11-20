#!/usr/bin/env bash
# Removes the launch agent that install.sh wrote, and only that file. It does not stop a running job: run the
# launchctl command this script prints for that.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="${STT_WATCH_AGENT_DIR:-$HOME/Library/LaunchAgents}"
case "$agent_dir" in /*) ;; *) echo "error: STT_WATCH_AGENT_DIR must be an absolute path" >&2; exit 2 ;; esac
plist="$agent_dir/com.sttstudio.watch.plist"

state="$(python3 "$HERE/plist.py" state "$plist")"
case "$state" in
    missing)
        echo "Nothing to remove: $plist does not exist."
        ;;
    generated)
        rm -f "$plist"
        echo "Removed $plist"
        ;;
    foreign)
        echo "error: $plist was not written by install.sh (no com.sttstudio.generated-by key). Left it in place." >&2
        exit 1
        ;;
esac

echo "If the agent is loaded, stop it with (this reports an error harmlessly when it is not loaded):"
printf '  launchctl bootout gui/%s/com.sttstudio.watch\n' "$(id -u)"
