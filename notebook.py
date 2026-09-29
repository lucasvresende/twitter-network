import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import networkx as nx


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _():
    ROOT_PATH = Path(__file__).resolve().parent
    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    return (TWITTER_COMBINED_TXT_PATH,)


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

    mo.ui.table(edges_lf)
    return (edges_lf,)


@app.cell
def _(edges_lf):
    followers_lf = (
        edges_lf.group_by("to")
        .agg(
            pl.col("from").count().alias("follower_count"),
            pl.col("from").implode().alias("followers"),
        )
        .rename({"to": "node"})
    )

    following_lf = (
        edges_lf.group_by("from")
        .agg(
            pl.col("to").count().alias("following_count"),
            pl.col("to").implode().alias("following"),
        )
        .rename({"from": "node"})
    )

    nodes_info_lf = (
        followers_lf.join(
            following_lf,
            on="node",
            how="full",
        )
        .with_columns(
            pl.col("followers").list.set_intersection("following").alias("friends"),
        )
        .with_columns(
            pl.col("friends").list.len().alias("friend_count"),
        )
        .select(
            pl.col("node").fill_null(pl.col("node_right")),
            pl.col("follower_count").fill_null(0),
            pl.col("following_count").fill_null(0),
            pl.col("friend_count").fill_null(0),
            pl.col("followers").fill_null([]),
            pl.col("following").fill_null([]),
            pl.col("friends").fill_null([]),
        )
    )

    mo.ui.table(nodes_info_lf)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convert Subset to Network
    """)
    return


@app.cell
def _(edges_lf):
    ego_nodes = (
        edges_lf.select(pl.col("from"))
        .unique(maintain_order=True)
        .collect()
        .to_series()
        .to_list()
    )

    ego_node_dropdown = mo.ui.dropdown(
        options=ego_nodes,
        value=ego_nodes[0],
        label="Choose an ego node",
    )

    ego_node_dropdown
    return (ego_node_dropdown,)


@app.cell
def _(edges_lf, ego_node_dropdown):
    ego_node = ego_node_dropdown.value

    ego_edges_lf = edges_lf.filter(
        (pl.col("from") == ego_node) | (pl.col("to") == ego_node)
    )
    return (ego_edges_lf,)


@app.cell
def _(ego_edges_lf):
    mo.ui.table(ego_edges_lf)
    return


if __name__ == "__main__":
    app.run()
