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
