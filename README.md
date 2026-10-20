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

Then open the notebook with `uv run marimo edit notebook.py`.
The `data/` directory is excluded from Git.

To use Codex in marimo's agent panel, start the bridge in a separate terminal
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
