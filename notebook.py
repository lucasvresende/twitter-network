import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path
    from collections.abc import Sequence, Generator

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import networkx as nx


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Edges
    """)
    return


@app.cell
def _():
    ROOT_PATH = Path(__file__).resolve().parent

    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    return DATA_PATH, TWITTER_COMBINED_TXT_PATH


@app.cell
def _(TWITTER_COMBINED_TXT_PATH):
    edges_lf = (
        pl.scan_csv(
            TWITTER_COMBINED_TXT_PATH,
            separator=" ",
            has_header=False,
            new_columns=("from", "to"),
        )
        .unique(["from", "to"])
        .sort(["from", "to"])
    )
    return (edges_lf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Features
    """)
    return


@app.cell
def _(DATA_PATH):
    TWITTER_EGO_FILES_PATH = DATA_PATH / "twitter"

    CIRCLES_FILES = list(TWITTER_EGO_FILES_PATH.rglob("*.circles"))
    EDGES_FILES = list(TWITTER_EGO_FILES_PATH.rglob("*.edges"))
    FEAT_FILES = list(TWITTER_EGO_FILES_PATH.rglob("*.feat"))
    FEATNAMES_FILES = list(TWITTER_EGO_FILES_PATH.rglob("*.featnames"))
    return (FEATNAMES_FILES,)


@app.cell
def _(FEATNAMES_FILES):
    _ego_nodes = [f.stem for f in FEATNAMES_FILES]

    _feats_items = [
        (
            pl.DataFrame(
                {
                    "raw_line": file_name.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                }
            )
            .lazy()
            .filter(pl.col("raw_line").str.strip_chars() != "")
            .select(
                pl.lit(node).cast(pl.Int64).alias("ego_node"),
                pl.col("raw_line")
                .str.extract(r"^\s*(\d+)\s+(.+?)\s*$", group_index=1)
                .cast(pl.Int64)
                .alias("index"),
                pl.col("raw_line")
                .str.extract(r"^\s*(\d+)\s+(.+?)\s*$", group_index=2)
                .alias("name"),
            )
        )
        for file_name, node in zip(FEATNAMES_FILES, _ego_nodes)
    ]

    feats_lf = pl.concat(_feats_items, how="vertical").sort("ego_node")

    feats_lf.collect()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gather Node Data
    """)
    return


@app.cell(hide_code=True)
def _(edges_lf):
    _followers_lf = (
        edges_lf.group_by("to")
        .agg(
            pl.col("from").count().alias("followers_count"),
            pl.col("from").implode().alias("followers"),
        )
        .rename({"to": "node"})
    )

    _following_lf = (
        edges_lf.group_by("from")
        .agg(
            pl.col("to").count().alias("following_count"),
            pl.col("to").implode().alias("following"),
        )
        .rename({"from": "node"})
    )

    # Data on followers, following, and friends for all nodes
    nodes_lf = (
        _followers_lf.join(
            _following_lf,
            on="node",
            how="full",
        )
        .with_columns(
            pl.col("followers").list.set_intersection("following").alias("friends"),
        )
        .with_columns(
            pl.col("friends").list.len().alias("friends_count"),
        )
        .select(
            pl.col("node").fill_null(pl.col("node_right")),
            pl.col(f"followers_count").fill_null(0),
            pl.col("following_count").fill_null(0),
            pl.col("friends_count").fill_null(0),
            pl.col("followers").fill_null([]),
            pl.col("following").fill_null([]),
            pl.col("friends").fill_null([]),
        )
    )

    mo.ui.table(nodes_lf)
    return (nodes_lf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Select Node To Analyze
    """)
    return


@app.cell(hide_code=True)
def _(nodes_lf):
    ego_nodes = nodes_lf.select("node").sort("node").collect().to_series().to_list()

    (
        ego_node_dropdown := mo.ui.dropdown(
            options=ego_nodes,
            value=ego_nodes[0],
            label="Choose an ego node",
        )
    )
    return (ego_node_dropdown,)


@app.cell(hide_code=True)
def _(ego_node_dropdown, nodes_lf):
    ego_node = ego_node_dropdown.value
    RELATIONSHIP_TYPES = ["followers", "following", "friends"]

    # The ego node + everyone the ego follows (Alters)
    ego_following_series = pl.Series([ego_node]).append(
        nodes_lf.filter(pl.col("node") == ego_node)
        .select(pl.col("following").explode())
        .collect()
        .to_series()
    )

    ### The LazyFrame with the network data
    egocentric_network_lf = (
        nodes_lf.filter(
            pl.col("node").is_in(ego_following_series),
        )
        .with_columns(
            pl.col(RELATIONSHIP_TYPES)
            .list.filter(pl.element().is_in(ego_following_series))
            .name.prefix("alter_")
        )
        .with_columns(
            pl.col("^alter_.*$").list.len().name.suffix("_count"),
        )
        .select(
            "node",
            *[
                pl.col([f"{rel}_count", f"{rel}", f"alter_{rel}_count", f"alter_{rel}"])
                for rel in RELATIONSHIP_TYPES
            ],
        )
    )

    mo.ui.table(egocentric_network_lf)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
