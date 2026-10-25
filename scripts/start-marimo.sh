#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

for dependency in uv node npm setsid; do
    command -v "$dependency" >/dev/null || {
        printf 'Required command not found: %s\n' "$dependency" >&2
        exit 1
    }
done

# Separate process groups let Ctrl+C stop both services and their children.
service_pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${service_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid bash "$script_dir/start-codex-agent.sh" &
service_pids+=("$!")

printf 'Opening marimo at http://localhost:2718/\n'
printf 'Select Codex in the agent panel and connect. Press Ctrl+C to stop both services.\n'
setsid uv run marimo edit notebook.py --watch --port 2718 --no-token &
service_pids+=("$!")

# If either service stops or fails, stop the other one too.
wait -n "${service_pids[@]}"
