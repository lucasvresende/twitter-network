# %%
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from twitter_network import paths


# %% feat
def load_feat(feat_path: Path, featnames_path: Path) -> pl.DataFrame:
    feats_df = pl.read_csv(feat_path, separator=" ", has_header=False).transpose()
    feats_df.columns = [str(col_name) for col_name in feats_df.row(0)]
    feats_df = feats_df.slice(1).cast(pl.Boolean)

    featnames_series = pl.read_csv(
        featnames_path,
        separator=" ",
        has_header=False,
        new_columns=("id", "featnames"),
    ).get_column("featnames")

    return feats_df.select(featnames_series, pl.all())


def filter_node_feats(feats_df: pl.DataFrame, node: int) -> pl.DataFrame:
    return feats_df.filter(str(node)).select(
        pl.lit(node).cast(pl.UInt64).alias("node"),
        pl.col("featnames"),
    )


# %%
def load_all_feats(
    feats_paths: Sequence[Path],
    featnames_paths: Sequence[Path],
) -> Sequence[pl.DataFrame]:
    dfs: list[pl.DataFrame] = []
    for feat_path, featnames_path in zip(feats_paths, featnames_paths):
        try:
            feats_df = load_feat(feat_path, featnames_path)
            for node in feats_df.columns[1:]:
                dfs.append(filter_node_feats(feats_df, int(node)))
            print(f"OK: {feat_path}")
        except pl.exceptions.ComputeError as e:
            print(f"ERR: {feat_path} - {e}. Skipping...")

    return dfs


# %% Normalize features and implode them into lists
def aggregate_feats(feats_dfs: Sequence[pl.DataFrame]) -> pl.LazyFrame:
    return (
        pl.concat(feats_dfs)
        .lazy()
        .with_columns(
            pl.col("featnames").str.to_lowercase().str.replace_all(r"[^a-z0-9@#]", "")
        )
        .unique()
        .group_by("node")
        .agg(
            pl.col("featnames").count().alias("feature_count"),
            pl.col("featnames").implode().alias("features"),
        )
    )


# %%
if __name__ == "__main__":
    n = 0
    feats_df = load_feat(paths.FEATS[n], paths.FEATNAMES[n])

    node_35415466_feats = filter_node_feats(feats_df, 35415466)
    node_189875309_feats = filter_node_feats(feats_df, 189875309)
    node_153226312_feats = filter_node_feats(feats_df, 153226312)

# %%
