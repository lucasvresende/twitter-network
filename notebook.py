# ruff: noqa: PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    # Standard-library utilities for paths, type hints, and atomic cache writes.
    from collections.abc import Generator, Sequence
    from pathlib import Path
    from tempfile import TemporaryDirectory

    # Notebook UI, lazy data processing, column selectors, and graph analysis.
    import marimo as mo
    import networkx as nx
    import polars as pl
    import polars.selectors as cs


@app.cell(hide_code=True)
def _():
    # Resolve every data location relative to this notebook so it can be run
    # from any working directory.
    ROOT_PATH = Path(__file__).resolve().parent

    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    PARQUET_OUT_PATH = DATA_PATH / "out"
    return DATA_PATH, PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Data

    The source dataset is loaded into Polars `LazyFrame` objects so transformations
    are deferred until a result is displayed, cached, or explicitly collected.
    Whenever a Parquet cache already exists, the notebook scans it instead of
    reparsing the original text files.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Edges

    Each row is a directed Twitter relationship from `from` to `to`. The loader
    prefers `data/out/edges.parquet`; otherwise it reads `twitter_combined.txt`,
    removes duplicate edges, and sorts the result for deterministic output.
    """)
    return


@app.cell(hide_code=True)
def _(PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH):
    # Reuse the normalized edge cache when available. On the first run, build the
    # same logical dataset lazily from the space-delimited source file.
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

    Twitter ego-network features are split across several file types:

    - `.featnames` maps an ego-specific feature index to its description.
    - `.feat` stores alter node IDs followed by binary feature vectors.
    - `.egofeat` stores the ego node's binary feature vector without a node ID.
    - `.circles` and `.edges` are discovered for possible later analysis but are
      not transformed in this section.

    Feature indices are local to an ego network, so `ego_node` is retained as part
    of every feature key.
    """)
    return


@app.cell(hide_code=True)
def _(DATA_PATH):
    # Discover every ego-network source file recursively. Sorting makes file
    # processing and resulting row order reproducible across operating systems.
    TWITTER_EGO_FILES_PATH = DATA_PATH / "twitter"

    CIRCLES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.circles"))
    EDGES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.edges"))
    FEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.feat"))
    EGOFEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.egofeat"))
    FEATNAMES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.featnames"))
    return EGOFEAT_FILES, FEATNAMES_FILES, FEAT_FILES, TWITTER_EGO_FILES_PATH


@app.cell(hide_code=True)
def _(FEATNAMES_FILES, PARQUET_OUT_PATH):
    # Produce one metadata row per ego-specific feature index:
    # (ego_node, index, name). Feature indices cannot be joined across ego networks
    # without including ego_node because each .featnames file defines its own map.
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
    # Normalize .feat and .egofeat records to one row per node:
    # (ego_node, node, is_ego, features). Keeping features as List[Int8] supports
    # different vector lengths across ego networks without creating sparse columns.
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

        # Read each source line as one string and tokenize its whitespace. For a
        # .feat row, the first token is the node ID; for .egofeat, the node ID is
        # the ego ID encoded in the filename.
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
    mo.md("""
    ## Save Data to Parquet

    This section converts the normalized lazy datasets into reusable Parquet
    caches under **`data/out`**. Missing caches are written with Zstandard
    compression; existing files are reused without being rewritten. Downstream
    cells always receive a fresh lazy scan of the Parquet file, so the application
    benefits from predicate and column pushdown.

    Cached datasets:

    - `edges.parquet`: deduplicated and sorted directed edges
    - `feature_names.parquet`: feature descriptions keyed by ego and index
    - `node_feature_vectors.parquet`: binary feature lists for alters and egos

    The expanded long-form feature table is intentionally not cached because it
    would duplicate the compact list representation.

    > To rebuild after changing source files or preprocessing logic, delete the
    > affected file from `data/out` and rerun the loading section.
    """)
    return


@app.function(hide_code=True)
def cache_lazyframe_to_parquet(
    frame: pl.LazyFrame,
    path: Path,
) -> tuple[pl.LazyFrame, dict]:
    """Cache a LazyFrame atomically and return its lazy scan plus a report.

    A new dataset is first written inside a temporary directory, validated by
    comparing schemas, and then moved into its final path. This prevents an
    interrupted write from leaving a partial cache. Existing caches are left
    untouched.
    """
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
    # Materialize each missing cache, then expose only Parquet-backed LazyFrames to
    # the rest of the notebook. The report makes cache state, row count, and disk
    # usage visible without collecting the full datasets into Python memory.
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

    The cached `node_feature_vectors_lf` keeps one compact binary list per node.
    For analysis that needs individual features, the next cell derives
    `node_features_lf` with one row per feature value and joins each local index to
    its description.

    Because this expanded view can be large, keep it lazy and filter by
    `ego_node`, `node`, or `value` before calling `collect()`. The displayed table
    below shows the compact vector dataset, not the expanded long-form result.
    """)
    return


@app.cell(hide_code=True)
def _(feats_lf, node_feature_vectors_lf):
    # Expand each binary vector into rows keyed by its zero-based feature index,
    # then attach the ego-specific feature description. This remains lazy; filters
    # applied downstream can be pushed into the Parquet scans where possible.
    node_features_lf = (
        node_feature_vectors_lf.with_columns(
            pl.int_ranges(
                0,
                pl.col("features").list.len(),
                dtype=pl.Int64,
            ).alias("index"),
        )
        .explode("index", "features")
        .rename({"features": "value"})
        .join(feats_lf, on=["ego_node", "index"], how="left")
        .select("ego_node", "node", "is_ego", "index", "name", "value")
        .sort(["ego_node", "node"])
    )

    # Display the compact cached representation to avoid rendering the much larger
    # exploded table by default.
    mo.ui.table(node_feature_vectors_lf)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gather Node Data

    The directed edge list is aggregated into one row per Twitter node. For each
    node, the next cell computes:

    - **followers**: nodes with an edge pointing to this node
    - **following**: nodes this node points to
    - **friends**: the intersection of followers and following (mutual edges)

    Both the member lists and their counts are retained. Nodes appearing on only
    one side of an edge are preserved by a full join, and missing lists/counts are
    replaced with empty lists and zeros.
    """)
    return


@app.cell(hide_code=True)
def _(edges_lf):
    # Aggregate incoming edges to obtain each node's followers.
    _followers_lf = (
        edges_lf.group_by("to")
        .agg(
            pl.col("from").count().alias("followers_count"),
            pl.col("from").implode().alias("followers"),
        )
        .rename({"to": "node"})
    )

    # Aggregate outgoing edges to obtain the accounts each node follows.
    _following_lf = (
        edges_lf.group_by("from")
        .agg(
            pl.col("to").count().alias("following_count"),
            pl.col("to").implode().alias("following"),
        )
        .rename({"from": "node"})
    )

    # Combine both directions and identify mutual relationships. The full join
    # preserves nodes that have only incoming or only outgoing edges.
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
            pl.col("followers_count").fill_null(0),
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
    ## Select Node to Analyze

    Choose an ego node to construct a reactive egocentric network. Changing the
    dropdown reruns only the dependent cells. This notebook defines the network as
    the selected ego plus every account the ego follows.

    The resulting table retains each node's global relationship lists and adds
    `alter_*` columns restricted to members of this selected network.
    """)
    return


@app.cell(hide_code=True)
def _(nodes_lf):
    # Materialize only the sorted node IDs needed as dropdown options. The selected
    # value is read in the next cell because marimo UI values cannot be consumed in
    # the same cell where the UI element is defined.
    ego_nodes = nodes_lf.select("node").sort("node").collect().to_series().to_list()

    ego_node_dropdown = mo.ui.dropdown(
        options=ego_nodes,
        value=ego_nodes[0],
        label="Choose an ego node",
    )

    ego_node_dropdown
    return (ego_node_dropdown,)


@app.cell(hide_code=True)
def _(ego_node_dropdown, nodes_lf):
    ego_node = ego_node_dropdown.value
    RELATIONSHIP_TYPES = ["followers", "following", "friends"]

    # Define membership as the selected ego plus every node the ego follows.
    _ego_following_series = pl.Series([ego_node]).append(
        nodes_lf.filter(pl.col("node") == ego_node)
        .select(pl.col("following").explode())
        .collect()
        .to_series()
    )

    # Keep global relationship columns for context, then derive alter_* lists and
    # counts containing only relationships whose endpoints belong to this network.
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
