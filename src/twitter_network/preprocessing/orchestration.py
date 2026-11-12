from collections.abc import Sequence
from pathlib import Path

import polars as pl

from twitter_network import paths
from twitter_network.preprocessing.edges import aggregate_relationships, load_edges_txt
from twitter_network.preprocessing.features import aggregate_feats, load_all_feats


def edges_and_features_to_polars(
    edges_file: Path,
    feats_paths: Sequence[Path],
    featnames_paths: Sequence[Path],
) -> pl.LazyFrame:
    edges_lf = load_edges_txt(edges_file)
    agg_nodes_lf = aggregate_relationships(edges_lf)
    feats_dfs = load_all_feats(feats_paths, featnames_paths)
    agg_feats_lf = aggregate_feats(feats_dfs)
    return agg_nodes_lf.join(agg_feats_lf, on="node")


if __name__ == "__main__":
    lf = edges_and_features_to_polars(
        edges_file=paths.TWITTER_COMBINED_TXT,
        feats_paths=paths.FEATS,
        featnames_paths=paths.FEATNAMES,
    )
    lf.sink_parquet(paths.DATA_OUT / "data.parquet")
