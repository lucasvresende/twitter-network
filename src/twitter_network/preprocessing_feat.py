# %%
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


# %% featnames


# %%
if __name__ == "__main__":
    feats_df = load_feat(paths.FEATS[0], paths.FEATNAMES[0])

    node_35415466_feats = filter_node_feats(feats_df, 35415466)

# %%
