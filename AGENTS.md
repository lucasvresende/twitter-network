# Working on the marimo notebook

## Start and connect

- Use the `/marimo-pair` skill when working on `notebook.py`. Read the installed
  skill's `SKILL.md` and use its bundled `execute-code.sh` to execute Python in
  the live notebook kernel.
- Reuse the running notebook at `http://localhost:2718/`. If it is not running,
  start it from the project root and keep its browser UI open:

  ```sh
  uv run marimo edit notebook.py --watch --port 2718 --no-token
  ```

- To use Codex in marimo's agent panel, run the project bridge in a separate
  terminal, then select **Codex** in the panel and connect:

  ```sh
  bash scripts/start-codex-agent.sh
  ```

  See `README.md` for bridge setup and troubleshooting.
- Start each code-mode session with the skill's required inspection command:

  ```sh
  bash /home/lvr/.agents/skills/marimo-pair/scripts/execute-code.sh \
    --url http://localhost:2718/ \
    -c 'import marimo._code_mode as cm; help(cm)'
  ```

  Resolve the script relative to the installed skill if its location differs.
  Wait for successful output before using `cm.get_context()` or other code-mode
  APIs. Use the same script and `--url` for subsequent notebook execution.

## Edit notebook cells

- Treat the live runtime as the source of truth. Make persistent notebook
  changes through `marimo._code_mode` in the scratchpad; do not directly edit
  the running notebook's `.py` file or import the private API in notebook cells.
- **Keep new cells' code hidden by default.** Pass `hide_code=True` explicitly
  to `ctx.create_cell(...)`, unless the user requests visible code. This
  collapses the editor; it does not disable the cell or hide its output.
- Give every new block a unique, descriptive name using `name=...`. Preserve
  existing code-visibility settings when editing cells.
- Read the current cell body before replacing it. Reuse existing imports and
  variables, and preserve marimo's dependency graph constraints.
- Creating or editing a cell does not by itself guarantee execution. Queue
  `ctx.run_cell(...)` when execution is needed, and inspect cell errors afterward.
- For this marimo version, programmatic dropdown updates through
  `ctx.set_ui_value(...)` take a one-item list such as `[label]`. The dropdown
  constructor's `value` remains the selected label itself.

## Validate changes

- Check relevant live cell errors and interactions after functional changes.
- Run `uv run marimo check notebook.py` and `git diff --check` before committing
  notebook edits. Keep unrelated user changes out of commits unless requested.
