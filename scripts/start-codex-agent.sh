#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

for dependency in node npm; do
    command -v "$dependency" >/dev/null || {
        printf 'Required command not found: %s\n' "$dependency" >&2
        exit 1
    }
done

# Marimo 0.25's launch hint uses the older @zed-industries adapter, which
# rejects current models. Install locked, project-local tools instead of
# depending on Zed's private installation or resolving npx packages each run.
adapter="${MARIMO_CODEX_ADAPTER:-$script_dir/node_modules/@agentclientprotocol/codex-acp/dist/index.js}"
bridge="$script_dir/node_modules/stdio-to-ws/dist/main.js"
if [[ ! -f "$adapter" || ! -f "$bridge" ]]; then
    npm ci --prefix "$script_dir" --no-audit --no-fund
fi
if [[ ! -f "$adapter" ]]; then
    printf 'Codex ACP adapter not found: %s\n' "$adapter" >&2
    exit 1
fi

# Keep the adapter's compatible bundled Codex unless explicitly overridden.
printf 'Starting Codex ACP bridge at ws://localhost:3021/message\n'
exec node "$bridge" "node \"$adapter\"" --port 3021
