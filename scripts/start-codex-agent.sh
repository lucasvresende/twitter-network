#!/usr/bin/env bash
set -euo pipefail

# Use the current ACP adapter installed by Zed. Marimo's bundled launch hint
# selects the older @zed-industries adapter, which rejects current models.
adapter="${MARIMO_CODEX_ADAPTER:-$HOME/.local/share/zed/external_agents/registry/npx/codex-acp/node_modules/@agentclientprotocol/codex-acp/dist/index.js}"
if [[ ! -f "$adapter" ]]; then
    printf 'Current Codex ACP adapter not found: %s\nSet MARIMO_CODEX_ADAPTER to its index.js path.\n' "$adapter" >&2
    exit 1
fi

export CODEX_PATH="${CODEX_PATH:-$(command -v codex)}"
exec npx --yes stdio-to-ws "node \"$adapter\"" --port 3021 --cors http://localhost:2718
