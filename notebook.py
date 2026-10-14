import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path
    from collections.abc import Sequence, Generator
    from tempfile import TemporaryDirectory

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import networkx as nx


@app.cell(hide_code=True)
def _():
    ROOT_PATH = Path(__file__).resolve().parent

    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    PARQUET_OUT_PATH = DATA_PATH / "out"
    return DATA_PATH, PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH


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


@app.cell(hide_code=True)
def _(PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH):
    _cache_path = PARQUET_OUT_PATH / "edges.parquet"
    if _cache_path.is_file():
        loaded_edges_lf = pl.scan_parquet(_cache_path)
    else:
        loaded_edges_lf = (
            pl.scan_csv(
                TWITTER_COMBINED_TXT_PATH,
                separator=" ",
                has_header=False,
                new_columns=("from", "to"),
            )
            .unique(["from", "to"])
            .sort(["from", "to"])
        )
    return (loaded_edges_lf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Features
    """)
    return


@app.cell(hide_code=True)
def _(DATA_PATH):
    TWITTER_EGO_FILES_PATH = DATA_PATH / "twitter"

    CIRCLES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.circles"))
    EDGES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.edges"))
    FEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.feat"))
    EGOFEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.egofeat"))
    FEATNAMES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.featnames"))
    return EGOFEAT_FILES, FEATNAMES_FILES, FEAT_FILES, TWITTER_EGO_FILES_PATH


@app.cell(hide_code=True)
def _(FEATNAMES_FILES, PARQUET_OUT_PATH):
    _cache_path = PARQUET_OUT_PATH / "feature_names.parquet"
    if _cache_path.is_file():
        loaded_feats_lf = pl.scan_parquet(_cache_path)
    else:
        if not FEATNAMES_FILES:
            raise FileNotFoundError(
                "No .featnames files found and no feature-name cache exists"
            )
        _regex = r"^\s*(\d+)\s+(.+?)\s*$"
        _featnames_items = [
            pl.DataFrame(
                {
                    "raw_line": _file.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                }
            )
            .lazy()
            .filter(pl.col("raw_line").str.strip_chars() != "")
            .select(
                pl.lit(int(_file.stem), dtype=pl.Int64).alias("ego_node"),
                pl.col("raw_line")
                .str.extract(_regex, group_index=1)
                .cast(pl.Int64)
                .alias("index"),
                pl.col("raw_line").str.extract(_regex, group_index=2).alias("name"),
            )
            for _file in FEATNAMES_FILES
        ]
        loaded_feats_lf = pl.concat(_featnames_items, how="vertical").sort(
            ["ego_node", "index"]
        )
    return (loaded_feats_lf,)


@app.cell(hide_code=True)
def _(EGOFEAT_FILES, FEAT_FILES, PARQUET_OUT_PATH, TWITTER_EGO_FILES_PATH):
    _cache_path = PARQUET_OUT_PATH / "node_feature_vectors.parquet"
    if _cache_path.is_file():
        loaded_node_feature_vectors_lf = pl.scan_parquet(_cache_path)
    else:
        _feature_paths = sorted([*FEAT_FILES, *EGOFEAT_FILES])
        if not _feature_paths:
            raise FileNotFoundError(
                f"No .feat or .egofeat files in {TWITTER_EGO_FILES_PATH}"
            )
        if any(not _path.stem.isdecimal() for _path in _feature_paths):
            raise ValueError("Feature filenames must have numeric ego-node IDs")

        # Tokenize whole lines to support varying vector lengths and whitespace.
        loaded_node_feature_vectors_lf = (
            pl.scan_csv(
                _feature_paths,
                has_header=False,
                separator="\x1f",
                quote_char=None,
                schema={"raw_line": pl.String},
                include_file_paths="source_file",
                raise_if_empty=False,
            )
            .filter(pl.col("raw_line").str.strip_chars() != "")
            .with_columns(
                pl.col("source_file")
                .str.extract(r"(\d+)\.(?:egofeat|feat)$", group_index=1)
                .cast(pl.Int64)
                .alias("ego_node"),
                pl.col("source_file").str.ends_with(".egofeat").alias("is_ego"),
                pl.col("raw_line").str.extract_all(r"\S+").alias("tokens"),
            )
            .select(
                "ego_node",
                pl.when(pl.col("is_ego"))
                .then(pl.col("ego_node"))
                .otherwise(pl.col("tokens").list.first().cast(pl.Int64))
                .alias("node"),
                "is_ego",
                pl.when(pl.col("is_ego"))
                .then(pl.col("tokens"))
                .otherwise(pl.col("tokens").list.slice(1))
                .cast(pl.List(pl.Int8))
                .alias("features"),
            )
        )
    return (loaded_node_feature_vectors_lf,)


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Save Data to Parquet

    Missing caches are created automatically in **`data/out`** using Zstandard
    compression. Existing caches are reused without rewriting them. Downstream
    analysis uses lazy Parquet scans, including on the first run after export.

    - `edges.parquet`: deduplicated, sorted edges
    - `feature_names.parquet`: feature names indexed by ego network
    - `node_feature_vectors.parquet`: feature lists for nodes and egos

    The expanded long-form feature table is not saved, to avoid duplicating data.

    **Refreshing the cache:** changes to source files or preprocessing are not
    tracked automatically. Delete the affected Parquet files from `data/out` and
    restart/run the notebook from the loading section to rebuild them.
    """)
    return


@app.function(hide_code=True)
def cache_lazyframe_to_parquet(frame: pl.LazyFrame, path: Path):
    """Create a missing cache safely, then return its lazy scan and status."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.is_file()
    if created:
        with TemporaryDirectory(prefix=".parquet-", dir=path.parent) as temp_dir:
            temporary_path = Path(temp_dir) / path.name
            frame.sink_parquet(temporary_path, compression="zstd")
            if (
                pl.scan_parquet(temporary_path).collect_schema()
                != frame.collect_schema()
            ):
                raise ValueError(f"Parquet schema mismatch for {path.name}")
            temporary_path.replace(path)

    cached_frame = pl.scan_parquet(path)
    report = {
        "file": path.name,
        "status": "Created" if created else "Reused",
        "rows": cached_frame.select(pl.len()).collect().item(),
        "size_mib": round(path.stat().st_size / (1024**2), 2),
    }
    return cached_frame, report


@app.cell(hide_code=True)
def _(
    PARQUET_OUT_PATH,
    loaded_edges_lf,
    loaded_feats_lf,
    loaded_node_feature_vectors_lf,
):
    edges_lf, _edges_cache_report = cache_lazyframe_to_parquet(
        loaded_edges_lf,
        PARQUET_OUT_PATH / "edges.parquet",
    )
    feats_lf, _feats_cache_report = cache_lazyframe_to_parquet(
        loaded_feats_lf,
        PARQUET_OUT_PATH / "feature_names.parquet",
    )
    node_feature_vectors_lf, _vectors_cache_report = cache_lazyframe_to_parquet(
        loaded_node_feature_vectors_lf,
        PARQUET_OUT_PATH / "node_feature_vectors.parquet",
    )

    parquet_cache_report = pl.DataFrame(
        [_edges_cache_report, _feats_cache_report, _vectors_cache_report]
    )
    mo.ui.table(parquet_cache_report, selection=None)
    return edges_lf, feats_lf, node_feature_vectors_lf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Consolidated Feature View
    """)
    return


@app.cell
def _(feats_lf, node_feature_vectors_lf):
    # Optional long-form view: one row per (ego_node, node, feature index).
    # Filter by ego_node/node before collecting this potentially large table.
    node_features_lf = (
        node_feature_vectors_lf.with_columns(
            pl.int_ranges(0, pl.col("features").list.len(), dtype=pl.Int64).alias(
                "index"
            )
        )
        .explode("index", "features")
        .rename({"features": "value"})
        .join(feats_lf, on=["ego_node", "index"], how="left")
        .select("ego_node", "node", "is_ego", "index", "name", "value")
    )

    mo.ui.table(node_feature_vectors_lf.head(10).collect(), selection=None)
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


@app.cell
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
    _ego_following_series = pl.Series([ego_node]).append(
        nodes_lf.filter(pl.col("node") == ego_node)
        .select(pl.col("following").explode())
        .collect()
        .to_series()
    )

    ### The LazyFrame with the network data
    egocentric_network_lf = (
        nodes_lf.filter(
            pl.col("node").is_in(_ego_following_series),
        )
        .with_columns(
            pl.col(RELATIONSHIP_TYPES)
            .list.filter(pl.element().is_in(_ego_following_series))
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


if __name__ == "__main__":
    app.run()
