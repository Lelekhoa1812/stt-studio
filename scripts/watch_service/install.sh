#!/usr/bin/env bash
# Writes the launch agent that runs scripts/watch_folder.py in the background. It does not load the agent:
# run the launchctl command this script prints when you want it started.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: install.sh WATCH_DIR OUTPUT_DIR [--format txt|srt|vtt|json] [--model NAME] [--language CODE]
                  [--diarize] [--interval SECONDS] [--url URL]

Writes ~/Library/LaunchAgents/com.sttstudio.watch.plist (or $STT_WATCH_AGENT_DIR). Both folders must be
absolute, and OUTPUT_DIR must not be inside WATCH_DIR. Nothing is loaded or started.
EOF
}

fail() {
    echo "error: $*" >&2
    exit 2
}

format=srt
model=""
language=""
diarize=0
interval=10
url=""
watch_arg=""
output_arg=""
positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --format | --model | --language | --interval | --url)
            [ $# -ge 2 ] || fail "$1 needs a value"
            case "$1" in
                --format) format="$2" ;;
                --model) model="$2" ;;
                --language) language="$2" ;;
                --interval) interval="$2" ;;
                --url) url="$2" ;;
            esac
            shift 2
            ;;
        --diarize)
            diarize=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*) fail "unknown option: $1 (see --help)" ;;
        *)
            positional=$((positional + 1))
            if [ "$positional" -eq 1 ]; then
                watch_arg="$1"
            elif [ "$positional" -eq 2 ]; then
                output_arg="$1"
            else
                fail "too many arguments (see --help)"
            fi
            shift
            ;;
    esac
done
if [ "$positional" -ne 2 ]; then
    usage >&2
    exit 2
fi

agent_dir="${STT_WATCH_AGENT_DIR:-$HOME/Library/LaunchAgents}"
log_dir="${STT_WATCH_LOG_DIR:-$HOME/Library/Logs}"
case "$agent_dir" in /*) ;; *) fail "STT_WATCH_AGENT_DIR must be an absolute path" ;; esac
case "$log_dir" in /*) ;; *) fail "STT_WATCH_LOG_DIR must be an absolute path" ;; esac
plist="$agent_dir/com.sttstudio.watch.plist"

options=(write --plist "$plist" --log-dir "$log_dir" --watch "$watch_arg" --output "$output_arg"
    --format "$format" --interval "$interval")
if [ -n "$model" ]; then options+=(--model "$model"); fi
if [ -n "$language" ]; then options+=(--language "$language"); fi
if [ "$diarize" -eq 1 ]; then options+=(--diarize); fi
if [ -n "$url" ]; then options+=(--url "$url"); fi

staged="$(python3 "$HERE/plist.py" "${options[@]}")"
trap 'rm -f "$staged"' EXIT

if ! plutil -lint "$staged"; then
    echo "error: the generated plist failed plutil -lint; the existing agent (if any) was not changed" >&2
    exit 1
fi
mv -f "$staged" "$plist"
trap - EXIT

uid="$(id -u)"
log="$log_dir/stt-studio-watch.log"
printf 'Wrote %s\n' "$plist"
printf 'Watching %s -> %s (%s, every %s s)\n' "$watch_arg" "$output_arg" "$format" "$interval"
printf 'Log: %s\n' "$log"
echo
echo "Not loaded and not running. To start it now and at every login, run:"
printf '  launchctl bootstrap gui/%s %q\n' "$uid" "$plist"
echo "To stop it later, run:"
printf '  launchctl bootout gui/%s/com.sttstudio.watch\n' "$uid"
echo "If the agent is already loaded, run the bootout line before the bootstrap line so the new settings apply."
