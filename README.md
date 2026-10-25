# twitter-network

An analysis of the Twitter SNAP social circles dataset

The notebook uses the combined edge list and ego-network files from
[Stanford SNAP](https://snap.stanford.edu/data/ego-Twitter.html).
To restore the dataset, run from the project root:

```sh
mkdir -p data
curl --fail --location --retry 3 https://snap.stanford.edu/data/twitter_combined.txt.gz --output data/twitter_combined.txt.gz
gzip --decompress --keep data/twitter_combined.txt.gz
curl --fail --location --retry 3 https://snap.stanford.edu/data/twitter.tar.gz --output data/twitter.tar.gz
tar --extract --gzip --file data/twitter.tar.gz --directory data
curl --fail --location --retry 3 https://snap.stanford.edu/data/readme-Ego.txt --output data/readme-Ego.txt
```

This creates `data/twitter_combined.txt` and extracts the `.featnames`, `.feat`,
`.egofeat`, `.circles`, and `.edges` files into `data/twitter/`. The notebook
creates its Parquet caches in `data/out/` on the first run.

The `data/` directory is excluded from Git.

To open marimo with the Codex bridge in one terminal, run from the project root
(requires `uv`, Node.js, npm, and Linux's `setsid`):

```sh
bash scripts/start-marimo.sh
```

This opens the notebook at `http://localhost:2718/` and starts the project-local
Codex bridge. Select **Codex** in the agent panel and connect; sign in if asked.
Press **Ctrl+C** to stop both services. Stop any existing marimo server on port
2718 or bridge on port 3021 before using this launcher.

To run the services separately, open the notebook with
`uv run marimo edit notebook.py --watch --port 2718 --no-token` and start the
bridge in another terminal
(requires Node.js and npm; installs the locked dependencies on first use):

```sh
bash /home/lvr/projects/development/twitter-network/scripts/start-codex-agent.sh
```

Keep it running, select **Codex** in the notebook's agent panel, and connect.
If asked, sign in with your ChatGPT account. For the notebook itself, run:

```sh
cd /home/lvr/projects/development/twitter-network && uv run marimo edit notebook.py --watch --port 2718
```

Use this bridge launcher instead of marimo 0.25's suggested
`npx stdio-to-ws "npx @zed-industries/codex-acp" --port 3021` command: that uses
the older adapter and can reject current Codex models. Only one bridge can use
port 3021; stop an existing bridge with Ctrl+C before starting another.
