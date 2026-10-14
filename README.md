# twitter-network

An analysis of the Twitter SNAP social circles dataset

The notebook uses the combined edge list from
[Stanford SNAP](https://snap.stanford.edu/data/ego-Twitter.html).
To restore the dataset, run from the project root:

```sh
mkdir -p data
curl --fail --location --retry 3 https://snap.stanford.edu/data/twitter_combined.txt.gz --output data/twitter_combined.txt.gz
gzip --decompress --keep data/twitter_combined.txt.gz
curl --fail --location --retry 3 https://snap.stanford.edu/data/readme-Ego.txt --output data/readme-Ego.txt
```

Then open the notebook with `uv run marimo edit notebook.py`.
The `data/` directory is excluded from Git.
