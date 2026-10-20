# ruff: noqa: PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt


@app.cell(hide_code=True)
def resolve_data_paths():
    # Resolve every data location relative to this notebook so it can be run
    # from any working directory.
    ROOT_PATH = Path(__file__).resolve().parent

    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    PARQUET_OUT_PATH = DATA_PATH / "out"
    return DATA_PATH, PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH


@app.cell(hide_code=True)
def describe_data_loading():
    mo.md(r"""
    ## Load Data

    The source dataset is loaded into Polars `LazyFrame` objects so transformations
    are deferred until a result is displayed, cached, or explicitly collected.
    Whenever a Parquet cache already exists, the notebook scans it instead of
    reparsing the original text files.
    """)
    return


@app.cell(hide_code=True)
def describe_directed_edges():
    mo.md(r"""
    ### Edges

    Each row is a directed Twitter relationship from `from` to `to`. The loader
    prefers `data/out/edges.parquet`; otherwise it reads `twitter_combined.txt`,
    removes duplicate edges, and sorts the result for deterministic output.
    """)
    return


@app.cell(hide_code=True)
def load_directed_edges(PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH):
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
def describe_ego_features():
    mo.md(r"""
    ### Features

    Twitter ego-network features are split across several file types:

    - `.featnames` maps an ego-specific feature index to its description.
    - `.feat` stores alter node IDs followed by binary feature vectors.
    - `.egofeat` stores the ego node's binary feature vector without a node ID.
    - `.circles` and `.edges` are discovered for possible later analysis but are
      not transformed in this section.

    Features are identified globally by name. Local ego/index pairs are retained
    only as provenance so each source vector can be mapped to global feature IDs.
    """)
    return


@app.cell(hide_code=True)
def discover_ego_network_files(DATA_PATH):
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
def load_feature_catalog(FEATNAMES_FILES, PARQUET_OUT_PATH):
    # Read either the global feature catalog or a legacy ego/index name map.
    # The save cell upgrades legacy metadata to the global catalog format.
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
def load_node_feature_vectors(
    EGOFEAT_FILES,
    FEAT_FILES,
    PARQUET_OUT_PATH,
    TWITTER_EGO_FILES_PATH,
):
    # Source vectors are temporary inputs. The persistent node cache stores
    # global feature-ID lists instead of separate ego-specific binary vectors.
    _cache_path = PARQUET_OUT_PATH / "nodes.parquet"
    _catalog_path = PARQUET_OUT_PATH / "feature_names.parquet"
    if (
        _cache_path.is_file()
        and _catalog_path.is_file()
        and pl.scan_parquet(_cache_path).collect_schema().get("features")
        == pl.List(pl.Int64)
        and "sources" in pl.scan_parquet(_catalog_path).collect_schema()
    ):
        loaded_node_feature_vectors_lf = None
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
def describe_parquet_caching():
    mo.md("""
    ## Save Data to Parquet

    This section converts the normalized lazy datasets into reusable Parquet
    caches under **`data/out`**. Missing caches are written with Zstandard
    compression; existing files are reused unless their schema needs an upgrade.
    Downstream cells receive a lazy scan of the Parquet file, so the application
    benefits from predicate and column pushdown.

    Cached datasets:

    - `edges.parquet`: deduplicated and sorted directed edges
    - `feature_names.parquet`: one row per distinct feature name, with a unique
      integer `id` and `sources`, a list of `{ego_node, index}` structs. IDs start
      at 0 in alphabetical name order.
    - `nodes.parquet`: relationship lists and a sorted `features` list of global
      feature IDs per node, saved in **Gather Node Data** below. A feature is
      included if it is active in any source network, including ego records.
      Nodes without observed active features have an empty list.

    Source binary vectors are read only when building or upgrading the node
    cache; a separate vector cache is no longer used.

    > To rebuild after changing source files or preprocessing logic, delete the
    > affected file from `data/out` and rerun the loading section. When rebuilding
    > the feature catalog, also delete `nodes.parquet` so its IDs stay aligned.
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
def cache_edges_and_feature_catalog(
    PARQUET_OUT_PATH,
    loaded_edges_lf,
    loaded_feats_lf,
):
    # Materialize each missing cache, then expose only Parquet-backed LazyFrames to
    # the rest of the notebook. The report makes cache state, row count, and disk
    # usage visible without collecting the full datasets into Python memory.
    edges_lf, _edges_cache_report = cache_lazyframe_to_parquet(
        loaded_edges_lf,
        PARQUET_OUT_PATH / "edges.parquet",
    )
    _features_path = PARQUET_OUT_PATH / "feature_names.parquet"
    _features_schema = loaded_feats_lf.collect_schema()
    _features_need_upgrade = "sources" not in _features_schema
    if _features_need_upgrade:
        _feature_origins = loaded_feats_lf.select("name", "ego_node", "index")
        if (
            _feature_origins.select(pl.any_horizontal(pl.all().is_null()).any())
            .collect()
            .item()
        ):
            raise ValueError("Feature names and source ego/index pairs cannot be null")
        if (
            _feature_origins.group_by("ego_node", "index")
            .agg(pl.col("name").n_unique().alias("names"))
            .filter(pl.col("names") > 1)
            .select(pl.len())
            .collect()
            .item()
        ):
            raise ValueError(
                "Each source ego/index pair must identify one feature name"
            )
        _features_to_cache = (
            _feature_origins.unique()
            .sort(["name", "ego_node", "index"])
            .group_by("name", maintain_order=True)
            .agg(pl.struct("ego_node", "index").alias("sources"))
            .sort("name")
            .with_row_index("id")
            .with_columns(pl.col("id").cast(pl.Int64))
            .select("id", "name", "sources")
        )
    else:
        _features_to_cache = loaded_feats_lf
    if _features_need_upgrade and _features_path.is_file():
        # Replace the old cache only after its upgraded copy is complete.
        with TemporaryDirectory(
            prefix=".parquet-", dir=_features_path.parent
        ) as _features_temp_dir:
            _features_temp_path = Path(_features_temp_dir) / _features_path.name
            cache_lazyframe_to_parquet(_features_to_cache, _features_temp_path)
            _features_temp_path.replace(_features_path)
    feats_lf, _feats_cache_report = cache_lazyframe_to_parquet(
        _features_to_cache,
        _features_path,
    )
    parquet_cache_report = pl.DataFrame([_edges_cache_report, _feats_cache_report])
    mo.ui.table(parquet_cache_report, selection=None)
    return edges_lf, feats_lf


@app.cell(hide_code=True)
def describe_consolidated_features():
    mo.md(r"""
    ### Consolidated Feature View

    `feats_lf` is the unique global feature catalog. Its `sources` list records
    the local indices used to decode source vectors. `node_feature_lists_lf`
    merges active features across every observation of a node, keeping one sorted
    list of unique feature IDs per node. A zero in one observation does not remove
    a feature observed as active elsewhere. Nodes with no active features remain
    present with an empty list.

    These compact lists are saved directly in `nodes.parquet` below. The displayed
    table shows the global feature catalog with its provenance.
    """)
    return


@app.cell(hide_code=True)
def consolidate_node_features(
    PARQUET_OUT_PATH,
    feats_lf,
    loaded_node_feature_vectors_lf,
):
    if loaded_node_feature_vectors_lf is None:
        node_feature_lists_lf = pl.scan_parquet(
            PARQUET_OUT_PATH / "nodes.parquet"
        ).select("node", "features")
    else:
        _feature_lookup = (
            feats_lf.select("id", "sources")
            .explode("sources")
            .unnest("sources")
            .rename({"id": "feature_id"})
        )
        _active_features = (
            loaded_node_feature_vectors_lf.select(
                "node",
                "ego_node",
                pl.col("features")
                .list.eval(pl.arg_where(pl.element() == 1))
                .cast(pl.List(pl.Int64))
                .alias("index"),
            )
            .explode("index")
            .filter(pl.col("index").is_not_null())
            .join(_feature_lookup, on=["ego_node", "index"], how="left", validate="m:1")
        )
        if (
            _active_features.select(pl.col("feature_id").is_null().any())
            .collect()
            .item()
        ):
            raise ValueError(
                "An active source feature is missing from the feature catalog"
            )
        node_feature_lists_lf = (
            loaded_node_feature_vectors_lf.select("node")
            .unique()
            .join(
                _active_features.group_by("node").agg(
                    pl.col("feature_id").unique().sort().alias("features")
                ),
                on="node",
                how="left",
            )
            .with_columns(
                pl.col("features").fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
            )
            .sort("node")
        )
    mo.ui.table(feats_lf)
    return (node_feature_lists_lf,)


@app.cell(hide_code=True)
def describe_node_relationships():
    mo.md(r"""
    ## Gather Node Data

    The directed edge list is aggregated into one row per Twitter node. For each
    node, the next cell computes:

    - **followers**: nodes with an edge pointing to this node
    - **following**: nodes this node points to
    - **friends**: the intersection of followers and following (mutual edges)
    - **features**: sorted global feature IDs observed across all ego networks

    Both the member lists and their counts are retained. Nodes appearing on only
    one side of an edge are preserved by a full join, and missing lists/counts are
    replaced with empty lists and zeros. Nodes found only in feature files are
    included too. The resulting table is cached in `data/out/nodes.parquet`.
    """)
    return


@app.cell(hide_code=True)
def consolidate_and_cache_nodes(
    PARQUET_OUT_PATH,
    edges_lf,
    loaded_node_feature_vectors_lf,
    node_feature_lists_lf,
):
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
    _relationships_lf = (
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

    _nodes_path = PARQUET_OUT_PATH / "nodes.parquet"
    if loaded_node_feature_vectors_lf is None:
        nodes_lf = pl.scan_parquet(_nodes_path)
    else:
        _nodes_to_cache = (
            _relationships_lf.join(
                node_feature_lists_lf, on="node", how="full", coalesce=True
            )
            .with_columns(
                pl.col("followers_count", "following_count", "friends_count").fill_null(
                    0
                ),
                pl.col("followers", "following", "friends", "features").fill_null(
                    pl.lit([], dtype=pl.List(pl.Int64))
                ),
            )
            .sort("node")
        )
        if _nodes_path.is_file():
            with TemporaryDirectory(
                prefix=".parquet-", dir=_nodes_path.parent
            ) as _nodes_temp_dir:
                _nodes_temp_path = Path(_nodes_temp_dir) / _nodes_path.name
                cache_lazyframe_to_parquet(_nodes_to_cache, _nodes_temp_path)
                _nodes_temp_path.replace(_nodes_path)
        nodes_lf, _nodes_cache_report = cache_lazyframe_to_parquet(
            _nodes_to_cache, _nodes_path
        )
    # The consolidated node cache now contains all persisted feature lists.
    (PARQUET_OUT_PATH / "node_feature_vectors.parquet").unlink(missing_ok=True)
    mo.ui.table(nodes_lf)
    return (nodes_lf,)


@app.cell
def degree_analysis_intro():
    mo.md(r"""
    ## 1. Degree distributions and connection inequality

    Compare **followers (in-degree)**, **following (out-degree)**, and
    **mutual connections** across every node in the consolidated dataset,
    including nodes with zero connections. These are observed relationships in
    this dataset, not complete Twitter account totals.

    The distribution views describe typical values and the long tail. The
    inequality views describe how connections are distributed among nodes.
    Mutual counts measure node incidences, rather than distinct mutual pairs.
    """)
    return


@app.function
def compute_degree_profile(values: np.ndarray) -> dict:
    """Summarize nonnegative integer counts, retaining zero-degree nodes.

    ECDF uses P(D <= k); inclusive CCDF uses P(D >= k), both divided by
    all nodes. Lorenz points are compressed at tied-degree boundaries,
    which preserves the exact piecewise-linear curve and its Gini area.
    An all-zero population has undefined connection shares and Gini.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Degree counts must be a nonempty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("Degree counts must be nonnegative integers")
    values = values.astype(np.int64)
    ordered = np.sort(values)
    size = ordered.size
    total = int(ordered.sum())
    degrees, counts = np.unique(ordered, return_counts=True)
    cumulative_nodes = counts.cumsum()
    population_share = np.r_[0.0, cumulative_nodes / size]
    connection_share = (
        np.r_[0.0, (degrees * counts).cumsum() / total]
        if total
        else np.full(degrees.size + 1, np.nan)
    )
    # Integer-aligned bins cover the complete range, including zero.
    bin_width = max(1, int(np.ceil((int(ordered[-1]) + 1) / 40)))
    bin_edges = np.arange(0, int(ordered[-1]) + bin_width + 1, bin_width)
    bin_counts, bin_edges = np.histogram(ordered, bins=bin_edges)
    top_shares = []
    for fraction in (0.001, 0.01, 0.05, 0.10, 0.20):
        top_nodes = max(1, int(np.ceil(size * fraction)))
        top_connections = int(ordered[-top_nodes:].sum())
        top_shares.append(
            {
                "top_fraction": fraction,
                "top_group": f"Top {fraction:.1%}"
                if fraction < 0.01
                else f"Top {fraction:.0%}",
                "nodes": top_nodes,
                "actual_node_share": top_nodes / size,
                "connections": top_connections,
                "connection_share": top_connections / total if total else None,
            }
        )
    return {
        "summary": {
            "nodes": size,
            "total": total,
            "mean": float(ordered.mean()),
            "std": float(ordered.std()),
            "median": float(np.median(ordered)),
            "p90": float(np.quantile(ordered, 0.90)),
            "p99": float(np.quantile(ordered, 0.99)),
            "maximum": int(ordered[-1]),
            "zero_nodes": int(counts[0]) if degrees[0] == 0 else 0,
            "zero_share": float(np.mean(ordered == 0)),
            "gini": float(1 - 2 * np.trapezoid(connection_share, population_share))
            if total
            else None,
        },
        "frequency": pl.DataFrame(
            {
                "degree": degrees,
                "nodes": counts,
                "ecdf": cumulative_nodes / size,
                "ccdf": (size - cumulative_nodes + counts) / size,
            }
        ),
        "histogram": pl.DataFrame(
            {
                "lower": bin_edges[:-1],
                "upper": bin_edges[1:],
                "nodes": bin_counts,
            }
        ),
        "lorenz": pl.DataFrame(
            {
                "population_share": population_share,
                "connection_share": connection_share,
            }
        ).fill_nan(None),
        "top_shares": pl.DataFrame(top_shares),
    }


@app.cell
def degree_analysis_data(nodes_lf):
    # Collect only scalar counts; relationship and feature lists stay in Parquet.
    degree_counts = nodes_lf.select(
        "node", "followers_count", "following_count", "friends_count"
    ).collect()
    mo.stop(
        degree_counts.is_empty(), mo.md("No nodes are available for degree analysis.")
    )

    degree_metrics = {
        "Followers (in-degree)": "followers_count",
        "Following (out-degree)": "following_count",
        "Mutual connections": "friends_count",
    }
    degree_profiles = {
        _label: compute_degree_profile(degree_counts[_column].to_numpy())
        for _label, _column in degree_metrics.items()
    }
    degree_summary = pl.DataFrame(
        [
            {"metric": _label, **_profile["summary"]}
            for _label, _profile in degree_profiles.items()
        ]
    )
    degree_top_shares = pl.concat(
        [
            _profile["top_shares"].with_columns(pl.lit(_label).alias("metric"))
            for _label, _profile in degree_profiles.items()
        ]
    )
    return (
        degree_counts,
        degree_metrics,
        degree_profiles,
        degree_summary,
        degree_top_shares,
    )


@app.cell
def degree_analysis_summary(degree_counts, degree_summary):
    mo.vstack(
        [
            mo.md(f"### Summary across {degree_counts.height:,} nodes"),
            mo.ui.table(
                degree_summary.select(
                    pl.col("metric").alias("Relationship"),
                    pl.col("mean").alias("Mean"),
                    pl.col("median").alias("Median"),
                    pl.col("p90").alias("P90"),
                    pl.col("p99").alias("P99"),
                    pl.col("maximum").alias("Maximum"),
                    pl.col("zero_nodes").alias("Zero-degree nodes"),
                    pl.col("zero_share").alias("Zero-degree share"),
                ),
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={
                    "Mean": "{:.2f}",
                    "Median": "{:,.1f}",
                    "P90": "{:,.1f}",
                    "P99": "{:,.1f}",
                    "Maximum": "{:,}",
                    "Zero-degree nodes": "{:,}",
                    "Zero-degree share": "{:.3%}",
                },
            ),
            mo.md(
                "Percentiles use linear interpolation. All rows use the same node population, including zeros."
            ),
        ]
    )
    return


@app.cell
def degree_analysis_selector(degree_metrics):
    degree_metric_selector = mo.ui.dropdown(
        options=list(degree_metrics),
        value="Followers (in-degree)",
        label="Relationship to explore",
    )
    degree_metric_selector
    return (degree_metric_selector,)


@app.cell
def degree_analysis_selection(degree_metric_selector, degree_profiles):
    selected_degree_metric = degree_metric_selector.value
    selected_degree_profile = degree_profiles[selected_degree_metric]
    selected_degree_summary = selected_degree_profile["summary"]
    return (
        selected_degree_metric,
        selected_degree_profile,
        selected_degree_summary,
    )


@app.cell
def degree_analysis_distributions(
    selected_degree_metric,
    selected_degree_profile,
    selected_degree_summary,
):
    _frequency = selected_degree_profile["frequency"]
    _ecdf_data = _frequency.select("degree", "ecdf")
    if _ecdf_data["degree"].min() > 0:
        _ecdf_data = pl.concat(
            [pl.DataFrame({"degree": [0], "ecdf": [0.0]}), _ecdf_data]
        )
    _tail = _frequency.filter(pl.col("degree") > 0)
    _blue = "#2563a6"

    degree_histogram_chart = (
        alt.Chart(selected_degree_profile["histogram"])
        .mark_bar(color=_blue, binSpacing=0)
        .encode(
            x=alt.X(
                "lower:Q",
                bin="binned",
                title="Connections per node",
                axis=alt.Axis(format=",d"),
            ),
            x2="upper:Q",
            y=alt.Y("nodes:Q", title="Number of nodes", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("lower:Q", title="From (inclusive)", format=",d"),
                alt.Tooltip("upper:Q", title="To (exclusive)", format=",d"),
                alt.Tooltip("nodes:Q", title="Nodes", format=",d"),
            ],
        )
        .properties(
            title=f"Degree histogram — {selected_degree_metric}",
            width="container",
            height=300,
        )
    )
    degree_ecdf_chart = (
        alt.Chart(_ecdf_data)
        .mark_line(color=_blue, interpolate="step-after")
        .encode(
            x=alt.X(
                "degree:Q", title="Connections per node (k)", scale=alt.Scale(zero=True)
            ),
            y=alt.Y(
                "ecdf:Q",
                title="Share of nodes with degree ≤ k",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("degree:Q", title="Degree", format=",d"),
                alt.Tooltip("ecdf:Q", title="Share ≤ k", format=".2%"),
            ],
        )
        .properties(
            title=f"Empirical cumulative distribution — {selected_degree_metric}",
            width="container",
            height=300,
        )
    )
    degree_ccdf_chart = (
        alt.Chart(_tail)
        .mark_line(color=_blue, interpolate="step-before", point=True)
        .encode(
            x=alt.X(
                "degree:Q",
                title="Connections per node (k, log scale)",
                axis=alt.Axis(values=[1, 10, 100, 1000, 10000], format="~s"),
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "ccdf:Q",
                title="Share with degree ≥ k (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(
                    values=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001], format=".3~%"
                ),
            ),
            tooltip=[
                alt.Tooltip("degree:Q", title="Degree", format=",d"),
                alt.Tooltip("ccdf:Q", title="Share ≥ k", format=".3%"),
            ],
        )
        .properties(
            title=f"Degree tail — {selected_degree_metric}",
            width="container",
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(f"### Distributions: {selected_degree_metric}"),
            mo.md(
                f"**Zero-degree nodes: {selected_degree_summary['zero_nodes']:,} ({selected_degree_summary['zero_share']:.3%}).** All statistics and probability denominators include these nodes."
            ),
            mo.ui.tabs(
                {
                    "Histogram": mo.vstack(
                        [
                            mo.md(
                                "Equal-width, integer-aligned bins cover the full observed range. Switch to the ECDF for exact cumulative shares."
                            ),
                            mo.ui.altair_chart(
                                degree_histogram_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                        ]
                    ),
                    "ECDF": mo.vstack(
                        [
                            mo.md(
                                "At k, the curve gives the fraction of all nodes with at most k connections."
                            ),
                            mo.ui.altair_chart(
                                degree_ecdf_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                        ]
                    ),
                    "Log-log tail": mo.vstack(
                        [
                            mo.md(
                                "Only positive degrees can appear on a log axis; shares still use all nodes as the denominator. The curve is P(D ≥ k), including equality. A straight-looking tail does not establish a power law."
                            ),
                            mo.ui.altair_chart(
                                degree_ccdf_chart,
                                chart_selection=False,
                                legend_selection=False,
                            )
                            if not _tail.is_empty()
                            else mo.md(
                                "No positive degrees are available for a log-log plot."
                            ),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def degree_analysis_inequality(
    degree_summary,
    degree_top_shares,
    selected_degree_metric,
    selected_degree_profile,
    selected_degree_summary,
):
    mo.stop(
        selected_degree_summary["total"] == 0,
        mo.md(
            "Connection shares, the Lorenz curve, and Gini are undefined when every node has zero connections."
        ),
    )
    _shares = selected_degree_profile["top_shares"]
    _top_one = _shares.filter(pl.col("top_fraction") == 0.01).row(0, named=True)
    _top_ten = _shares.filter(pl.col("top_fraction") == 0.10).row(0, named=True)
    _equality = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(color="#666666", strokeDash=[5, 5])
        .encode(
            x=alt.X("x:Q", title="Cumulative share of nodes (least to most connected)"),
            y=alt.Y("y:Q", title="Cumulative share of connections"),
        )
    )
    _lorenz = (
        alt.Chart(selected_degree_profile["lorenz"])
        .mark_line(color="#2563a6", strokeWidth=2)
        .encode(
            x=alt.X(
                "population_share:Q",
                title="Cumulative share of nodes (least to most connected)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5),
            ),
            y=alt.Y(
                "connection_share:Q",
                title="Cumulative share of connections",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5),
            ),
            tooltip=[
                alt.Tooltip("population_share:Q", title="Node share", format=".2%"),
                alt.Tooltip(
                    "connection_share:Q", title="Connection share", format=".2%"
                ),
            ],
        )
    )
    degree_lorenz_chart = (_lorenz + _equality).properties(
        title=f"Lorenz curve — {selected_degree_metric}", width="container", height=340
    )
    _share_bars = (
        alt.Chart(_shares)
        .mark_bar(color="#2563a6")
        .encode(
            x=alt.X(
                "connection_share:Q",
                title="Share of all connections",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5),
            ),
            y=alt.Y("top_group:N", title=None, sort=_shares["top_group"].to_list()),
            tooltip=[
                alt.Tooltip("top_group:N", title="Ranked group"),
                alt.Tooltip("nodes:Q", title="Nodes in group", format=",d"),
                alt.Tooltip(
                    "actual_node_share:Q", title="Actual node share", format=".3%"
                ),
                alt.Tooltip("connections:Q", title="Connections", format=",d"),
                alt.Tooltip(
                    "connection_share:Q", title="Connection share", format=".2%"
                ),
            ],
        )
    )
    degree_top_share_chart = (
        _share_bars
        + _share_bars.mark_text(align="left", dx=5, color="#333333").encode(
            text=alt.Text("connection_share:Q", format=".1%")
        )
    ).properties(
        title=f"Connection share held by top-ranked nodes — {selected_degree_metric}",
        width="container",
        height=230,
    )
    mo.vstack(
        [
            mo.md(f"### Connection inequality: {selected_degree_metric}"),
            mo.hstack(
                [
                    mo.stat(
                        f"{selected_degree_summary['gini']:.3f}",
                        label="Gini coefficient",
                        caption="0 = equal counts; closer to 1 = more unequal",
                    ),
                    mo.stat(
                        f"{_top_one['connection_share']:.1%}",
                        label="Connection share held by top 1%",
                        caption=f"{_top_one['nodes']:,} nodes",
                    ),
                    mo.stat(
                        f"{_top_ten['connection_share']:.1%}",
                        label="Connection share held by top 10%",
                        caption=f"{_top_ten['nodes']:,} nodes",
                    ),
                ],
                widths="equal",
                wrap=True,
            ),
            mo.ui.altair_chart(
                degree_lorenz_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "The dashed diagonal represents equal counts for every node. The blue Lorenz curve ranks nodes from least to most connected. Gini is twice the area between these curves, without finite-sample correction."
            ),
            mo.ui.altair_chart(
                degree_top_share_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "Top groups are nested, not additive. Each contains ceil(fraction × node count) nodes ranked independently by the selected relationship; ties at the boundary have equal counts. Hover for actual group sizes."
            ),
            mo.accordion(
                {
                    "Exact concentration statistics for all relationships": mo.ui.table(
                        degree_summary.select("metric", "nodes", "total", "gini")
                        .join(degree_top_shares, on="metric", how="left", suffix="_top")
                        .select(
                            "metric",
                            "gini",
                            "top_group",
                            pl.col("nodes_top").alias("top_nodes"),
                            "actual_node_share",
                            "connections",
                            "connection_share",
                        ),
                        selection=None,
                        pagination=False,
                        show_data_types=False,
                        show_column_summaries=False,
                        format_mapping={
                            "gini": "{:.4f}",
                            "actual_node_share": "{:.3%}",
                            "connection_share": "{:.2%}",
                            "connections": "{:,}",
                            "top_nodes": "{:,}",
                        },
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def describe_ego_network_selection():
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
def select_ego_node(nodes_lf):
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
def build_egocentric_network(ego_node_dropdown, nodes_lf):
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
            "features",
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
