# %% Imports

from pathlib import Path

import polars as pl

from twitter_network import paths


# %% Functions
def load_edges_txt(file_path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            file_path,
            separator=" ",
            has_header=False,
            new_columns=("from", "to"),
        )
        .unique(["from", "to"])
        .sort(["from", "to"])
    )


def aggregate_followers(edges_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        edges_lf.group_by("to")
        .agg(
            pl.col("from").count().alias("follower_count"),
            pl.col("from").implode().alias("followers"),
        )
        .rename({"to": "node"})
    )


def aggregate_following(edges_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        edges_lf.group_by("from")
        .agg(
            pl.col("to").count().alias("following_count"),
            pl.col("to").implode().alias("following"),
        )
        .rename({"from": "node"})
    )


def aggregate_friends(edges_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        edges_lf.join(edges_lf, left_on=("from", "to"), right_on=("to", "from"))
        .group_by("from")
        .agg(
            pl.col("to").count().alias("friend_count"),
            pl.col("to").implode().alias("friends"),
        )
        .rename({"from": "node"})
    )


def aggregate_all(edges_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        edges_lf.pipe(aggregate_followers)
        .join(aggregate_following(edges_lf), on="node")
        .join(aggregate_friends(edges_lf), on="node")
    )


# %%
if __name__ == "__main__":
    lf = load_edges_txt(paths.TWITTER_COMBINED_TXT).pipe(aggregate_all)

# %%
