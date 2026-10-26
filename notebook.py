# ruff: noqa: PLR1711

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import networkx as nx
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits
    from hashlib import sha256


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

    Names are lowercased as soon as they are extracted, then every character other
    than a Unicode letter or number is removed. For example, `#NASA`, `@NASA:` and
    `#nasa` all become `nasa`. Names that become empty are discarded.

    Features with the same normalized name share one global code. Local ego/index
    pairs are retained as provenance so all original positions decode to that code.
    Membership is the union of matching positions and observations.
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
def define_feature_name_normalization():
    def normalize_feature_name(expression):
        """Lowercase names and retain only Unicode letters and numbers."""
        return expression.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]", "")

    def build_global_feature_catalog(origins):
        """Assign one deterministic code per nonempty normalized name."""
        if origins.select(pl.any_horizontal(pl.all().is_null()).any()).collect().item():
            raise ValueError("Feature names and source ego/index pairs cannot be null")
        if (
            origins.group_by("ego_node", "index")
            .agg(pl.col("name").n_unique().alias("names"))
            .filter(pl.col("names") > 1)
            .select(pl.len())
            .collect()
            .item()
        ):
            raise ValueError(
                "Each source ego/index pair must identify one normalized name"
            )
        return (
            origins.filter(pl.col("name") != "")
            .unique()
            .sort(["name", "ego_node", "index"])
            .group_by("name", maintain_order=True)
            .agg(pl.struct("ego_node", "index").alias("sources"))
            .sort("name")
            .with_row_index("id")
            .with_columns(pl.col("id").cast(pl.Int64))
            .select("id", "name", "sources")
            .collect()
        )

    return build_global_feature_catalog, normalize_feature_name


@app.cell(hide_code=True)
def load_feature_catalog(
    FEATNAMES_FILES,
    PARQUET_OUT_PATH,
    normalize_feature_name,
):
    # Normalize names at extraction, before grouping or assigning global codes.
    # Reading the small name files also retains source positions whose names
    # become empty, so those positions can be deliberately ignored during decoding.
    _cache_path = PARQUET_OUT_PATH / "feature_names.parquet"
    if FEATNAMES_FILES:
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
                normalize_feature_name(
                    pl.col("raw_line").str.extract(_regex, group_index=2)
                ).alias("name"),
            )
            for _file in FEATNAMES_FILES
        ]
        loaded_feats_lf = pl.concat(_featnames_items, how="vertical").sort(
            ["ego_node", "index"]
        )
    elif _cache_path.is_file():
        # A cached catalog can still support analysis without the source name files.
        _cached_catalog = pl.scan_parquet(_cache_path)
        loaded_feats_lf = (
            _cached_catalog.select("name", "sources")
            .explode("sources")
            .unnest("sources")
            if "sources" in _cached_catalog.collect_schema()
            else _cached_catalog.select("name", "ego_node", "index")
        ).with_columns(normalize_feature_name(pl.col("name")).alias("name"))
    else:
        raise FileNotFoundError(
            "No .featnames files found and no feature-name cache exists"
        )
    return (loaded_feats_lf,)


@app.cell(hide_code=True)
def load_node_feature_vectors(
    EGOFEAT_FILES,
    FEAT_FILES,
    PARQUET_OUT_PATH,
    TWITTER_EGO_FILES_PATH,
    feature_cache_metadata,
):
    # Source vectors are temporary inputs. The persistent node cache stores
    # global feature-ID lists instead of separate ego-specific binary vectors.
    _cache_path = PARQUET_OUT_PATH / "nodes.parquet"
    _node_metadata = (
        pl.read_parquet_metadata(_cache_path) if _cache_path.is_file() else {}
    )
    if (
        _cache_path.is_file()
        and pl.scan_parquet(_cache_path).collect_schema().get("features")
        == pl.List(pl.Int64)
        and all(
            _node_metadata.get(_key) == _value
            for _key, _value in feature_cache_metadata.items()
        )
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
    compression; existing feature and node files are reused only when their normalization
    version and feature-codebook digest match.
    Downstream cells receive a lazy scan of the Parquet file, so the application
    benefits from predicate and column pushdown.

    Cached datasets:

    - `edges.parquet`: deduplicated and sorted directed edges
    - `feature_names.parquet`: one row per distinct nonempty normalized feature name, with a unique
      integer `id` and `sources`, a list of `{ego_node, index}` structs. IDs start
      at 0 in alphabetical name order.
    - `nodes.parquet`: relationship lists and a sorted `features` list of global
      feature IDs per node, saved in **Gather Node Data** below. A feature is
      included if it is active in any source network, including ego records.
      Nodes without observed active features have an empty list.

    Source name files are normalized before building the codebook. Source binary
    vectors are read only when building or upgrading the node cache; a separate
    vector cache is no longer used. Feature and node Parquet metadata record the
    same normalization version and codebook digest. Old caches rebuild automatically,
    including after an interrupted rebuild, so node codes stay aligned.

    > Changes to source names update the codebook and rebuild dependent node codes
    > automatically. After changing binary vector or edge files, delete the affected
    > node or edge cache in `data/out` and rerun the loading section.
    """)
    return


@app.function(hide_code=True)
def cache_lazyframe_to_parquet(
    frame: pl.LazyFrame,
    path: Path,
    *,
    metadata: dict[str, str] | None = None,
) -> tuple[pl.LazyFrame, dict]:
    """Atomically create or refresh a cache whose requested metadata has changed.

    Feature and node caches share a normalization version and codebook digest.
    If rebuilding stops between the two files, their mismatched metadata forces
    node features to be decoded again rather than reusing incompatible codes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file()
    needs_write = not existed
    if existed and metadata is not None:
        cached_metadata = pl.read_parquet_metadata(path)
        needs_write = any(
            cached_metadata.get(key) != value for key, value in metadata.items()
        )
    if needs_write:
        with TemporaryDirectory(prefix=".parquet-", dir=path.parent) as temp_dir:
            temporary_path = Path(temp_dir) / path.name
            frame.sink_parquet(
                temporary_path, compression="zstd", metadata=metadata
            )
            if (
                pl.scan_parquet(temporary_path).collect_schema()
                != frame.collect_schema()
            ):
                raise ValueError(f"Parquet schema mismatch for {path.name}")
            if metadata is not None:
                written_metadata = pl.read_parquet_metadata(temporary_path)
                if any(
                    written_metadata.get(key) != value
                    for key, value in metadata.items()
                ):
                    raise ValueError(f"Parquet metadata mismatch for {path.name}")
            temporary_path.replace(path)
    cached_frame = pl.scan_parquet(path)
    report = {
        "file": path.name,
        "status": "Rebuilt"
        if needs_write and existed
        else "Created"
        if needs_write
        else "Reused",
        "rows": cached_frame.select(pl.len()).collect().item(),
        "size_mib": round(path.stat().st_size / (1024**2), 2),
    }
    return cached_frame, report


@app.cell(hide_code=True)
def cache_edges_and_feature_catalog(
    PARQUET_OUT_PATH,
    build_global_feature_catalog,
    loaded_edges_lf,
    loaded_feats_lf,
):
    edges_lf, _edges_cache_report = cache_lazyframe_to_parquet(
        loaded_edges_lf, PARQUET_OUT_PATH / "edges.parquet"
    )
    _normalized_catalog = build_global_feature_catalog(
        loaded_feats_lf.select("name", "ego_node", "index")
    )
    # Include both names/codes and source positions in the cache identity.
    # JSON serialization is deterministic because every level was sorted above.
    feature_cache_metadata = {
        "feature_normalization": "unicode-alnum-lower-v1",
        "feature_codebook_sha256": sha256(
            _normalized_catalog.write_json().encode("utf-8")
        ).hexdigest(),
    }
    feats_lf, _feats_cache_report = cache_lazyframe_to_parquet(
        _normalized_catalog.lazy(),
        PARQUET_OUT_PATH / "feature_names.parquet",
        metadata=feature_cache_metadata,
    )
    parquet_cache_report = pl.DataFrame([_edges_cache_report, _feats_cache_report])
    mo.ui.table(parquet_cache_report, selection=None)
    return edges_lf, feats_lf, feature_cache_metadata


@app.cell(hide_code=True)
def describe_consolidated_features():
    mo.md(r"""
    ### Consolidated Feature View

    `feats_lf` is the unique global catalog of lowercase, alphanumeric names. Its `sources` list records
    the local indices used to decode source vectors. `node_feature_lists_lf`
    merges matching source names and active features across every observation of a
    node, keeping one sorted
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
    loaded_feats_lf,
    loaded_node_feature_vectors_lf,
):
    if loaded_node_feature_vectors_lf is None:
        node_feature_lists_lf = pl.scan_parquet(
            PARQUET_OUT_PATH / "nodes.parquet"
        ).select("node", "features")
    else:
        # Multiple local positions may map to the same normalized global code.
        # Keep empty names in this lookup to distinguish deliberately discarded
        # annotations from source positions that are actually missing or malformed.
        _feature_lookup = (
            loaded_feats_lf.select("ego_node", "index", "name")
            .unique()
            .join(feats_lf.select("id", "name"), on="name", how="left", validate="m:1")
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
            .collect()
            .lazy()
        )
        if (
            _active_features.select(
                (
                    pl.col("name").is_null()
                    | ((pl.col("name") != "") & pl.col("feature_id").is_null())
                ).any()
            )
            .collect()
            .item()
        ):
            raise ValueError(
                "An active source feature is missing from the normalized catalog"
            )
        node_feature_lists_lf = (
            loaded_node_feature_vectors_lf.select("node")
            .unique()
            .join(
                _active_features.filter(pl.col("name") != "")
                .group_by("node")
                .agg(pl.col("feature_id").unique().sort().alias("features")),
                on="node",
                how="left",
            )
            .with_columns(
                pl.col("features").fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
            )
            .sort("node")
        )
    mo.ui.table(feats_lf, selection=None)
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
    feature_cache_metadata,
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
        nodes_lf, _nodes_cache_report = cache_lazyframe_to_parquet(
            _nodes_to_cache, _nodes_path, metadata=feature_cache_metadata
        )
    # The consolidated node cache now contains all persisted feature lists.
    (PARQUET_OUT_PATH / "node_feature_vectors.parquet").unlink(missing_ok=True)
    mo.ui.table(nodes_lf, selection=None)
    return (nodes_lf,)


@app.cell(hide_code=True)
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


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.function(hide_code=True)
def topic_widget_grid(items):
    """Arrange related widgets in equal responsive columns with readable spacing."""
    cards = "".join(
        '<div style="min-width:0;padding:1rem;border:1px solid '
        'var(--gray-5, #e2e8f0);border-radius:0.65rem;">'
        + mo.as_html(item).text
        + "</div>"
        for item in items
    )
    return mo.Html(
        '<div class="topic-widget-grid" style="display:grid;'
        "grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr));"
        'gap:1rem;align-items:stretch;width:100%;">' + cards + "</div>"
    )


@app.cell(hide_code=True)
def degree_analysis_selector(degree_metrics):
    degree_metric_selector = mo.ui.dropdown(
        options=list(degree_metrics),
        value="Followers (in-degree)",
        label="Relationship to explore",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("### Degree controls"),
            topic_widget_grid([degree_metric_selector]),
        ]
    )
    return (degree_metric_selector,)


@app.cell(hide_code=True)
def degree_analysis_selection(degree_metric_selector, degree_profiles):
    selected_degree_metric = degree_metric_selector.value
    selected_degree_profile = degree_profiles[selected_degree_metric]
    selected_degree_summary = selected_degree_profile["summary"]
    return (
        selected_degree_metric,
        selected_degree_profile,
        selected_degree_summary,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
        height=340,
    )
    mo.vstack(
        [
            mo.md(f"### Connection inequality: {selected_degree_metric}"),
            topic_widget_grid(
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
                ]
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                degree_lorenz_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                            mo.md(
                                "The dashed diagonal represents equal counts for every node. The blue Lorenz curve ranks nodes from least to most connected. Gini is twice the area between these curves, without finite-sample correction."
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                degree_top_share_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                            mo.md(
                                "Top groups are nested, not additive. Each contains ceil(fraction × node count) nodes ranked independently by the selected relationship; ties at the boundary have equal counts. Hover for actual group sizes."
                            ),
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.5,
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
def describe_node_centrality():
    mo.md(r"""
    ## 2. Node centrality

    Degree measures direct connections. **PageRank** measures attention received
    from accounts that themselves receive attention; **HITS authority** rewards
    being followed by strong hubs, while **HITS hub** rewards following strong
    authorities. These are structural scores, not evidence of real-world influence.

    An edge points from the follower to the followed account. All consolidated
    nodes are included; duplicate edges and self-loops are excluded, matching the
    connectivity analysis. In/out degree centrality divides each count by N − 1.
    PageRank uses damping 0.85, uniform teleportation, and redistributes dangling
    nodes' mass uniformly. HITS scores are normalized to sum to one. Scores from
    different methods have different scales: compare their ranks.
    """)
    return


@app.function(hide_code=True)
def compute_node_centrality(
    adjacency, node_ids, tolerance=1e-10, max_iterations=1000
):
    """Compute sparse directed centralities and record convergence diagnostics."""
    graph = adjacency.astype(np.float64).tocsr(copy=True)
    n = len(node_ids)
    if n == 0:
        raise ValueError("Centrality requires at least one node")
    outgoing = np.asarray(graph.sum(axis=1)).ravel()
    incoming = np.asarray(graph.sum(axis=0)).ravel()
    transpose = graph.T.tocsr()
    inverse_outgoing = np.divide(1.0, outgoing, out=np.zeros(n), where=outgoing > 0)
    pagerank = np.full(n, 1.0 / n)
    diagnostics = []
    for iteration in range(1, max_iterations + 1):
        updated = (
            0.85
            * (
                transpose @ (pagerank * inverse_outgoing)
                + pagerank[outgoing == 0].sum() / n
            )
            + 0.15 / n
        )
        residual = float(np.abs(updated - pagerank).sum())
        pagerank = updated
        if residual <= tolerance:
            break
    diagnostics.append(
        {
            "method": "PageRank",
            "iterations": iteration,
            "L1_change": residual,
            "converged": residual <= tolerance,
        }
    )
    hubs = np.full(n, 1.0 / n)
    authorities = np.zeros(n)
    residual = 0.0
    iteration = 0
    if graph.nnz:
        for iteration in range(1, max_iterations + 1):
            new_authorities = transpose @ hubs
            new_authorities /= new_authorities.sum()
            new_hubs = graph @ new_authorities
            new_hubs /= new_hubs.sum()
            residual = float(
                max(
                    np.abs(new_hubs - hubs).sum(),
                    np.abs(new_authorities - authorities).sum(),
                )
            )
            hubs, authorities = new_hubs, new_authorities
            if residual <= tolerance:
                break
    else:
        hubs = np.zeros(n)
    diagnostics.append(
        {
            "method": "HITS",
            "iterations": iteration,
            "L1_change": residual,
            "converged": residual <= tolerance,
        }
    )
    scores = pl.DataFrame(
        {
            "node": node_ids,
            "in_degree": incoming.astype(np.int64),
            "out_degree": outgoing.astype(np.int64),
            "In-degree centrality": incoming / max(n - 1, 1),
            "Out-degree centrality": outgoing / max(n - 1, 1),
            "PageRank": pagerank / pagerank.sum(),
            "HITS authority": authorities,
            "HITS hub": hubs,
        }
    )
    return scores, pl.DataFrame(diagnostics)


@app.cell(hide_code=True)
def compute_node_centrality_scores(network_adjacency, network_node_ids):
    centrality_scores, centrality_diagnostics = compute_node_centrality(
        network_adjacency, network_node_ids
    )
    centrality_methods = [
        "In-degree centrality",
        "Out-degree centrality",
        "PageRank",
        "HITS authority",
        "HITS hub",
    ]
    centrality_ranks = centrality_scores.select(
        "node",
        *[
            pl.col(_method).rank(method="average", descending=True).alias(_method)
            for _method in centrality_methods
        ],
    )
    centrality_rank_agreement = pl.DataFrame(
        [
            {
                "method": _first,
                "comparison": _second,
                "Spearman correlation": float(
                    np.corrcoef(
                        centrality_ranks[_first].to_numpy(),
                        centrality_ranks[_second].to_numpy(),
                    )[0, 1]
                ),
            }
            for _first in centrality_methods
            for _second in centrality_methods
        ]
    )
    mo.stop(
        not centrality_diagnostics["converged"].all(),
        mo.md(
            "Centrality iteration did not converge. Inspect centrality_diagnostics before interpreting rankings."
        ),
    )
    return (
        centrality_diagnostics,
        centrality_methods,
        centrality_rank_agreement,
        centrality_scores,
    )


@app.cell(hide_code=True)
def select_centrality_measure(centrality_methods):
    centrality_method_selector = mo.ui.dropdown(
        options=centrality_methods,
        value="PageRank",
        label="Centrality measure",
        full_width=True,
    )
    centrality_top_count = mo.ui.slider(
        start=10, stop=50, step=10, value=20, label="Top nodes", full_width=True
    )
    mo.vstack(
        [
            mo.md("### Centrality controls"),
            topic_widget_grid([centrality_method_selector, centrality_top_count]),
        ]
    )
    return centrality_method_selector, centrality_top_count


@app.cell(hide_code=True)
def display_node_centrality(
    centrality_method_selector,
    centrality_scores,
    centrality_top_count,
):
    _method = centrality_method_selector.value
    _ordered = centrality_scores.sort([_method, "node"], descending=[True, False])
    _top = _ordered.head(centrality_top_count.value).with_columns(
        pl.col("node").cast(pl.String).alias("account")
    )
    _values = _ordered[_method].to_numpy()
    _positive = _values[_values > 0]
    _top_fraction = (
        float(
            _values[: max(1, int(np.ceil(len(_values) * 0.01)))].sum() / _values.sum()
        )
        if _values.sum()
        else 0.0
    )
    _bar = (
        alt.Chart(_top)
        .mark_bar(color="#2563a6")
        .encode(
            x=alt.X(f"{_method}:Q", title=_method),
            y=alt.Y("account:N", sort=_top["account"].to_list(), title="Node ID"),
            tooltip=[
                "account:N",
                alt.Tooltip(f"{_method}:Q", format=".6g"),
                "in_degree:Q",
                "out_degree:Q",
            ],
        )
        .properties(
            width="container",
            height=centrality_top_count.value * 20,
            title=f"Top nodes by {_method}",
        )
    )
    _distribution_outputs = []
    if _positive.size:
        _logs = np.log10(_positive)
        _counts, _bins = np.histogram(_logs, bins=40)
        _histogram = pl.DataFrame(
            {"lower": 10.0 ** _bins[:-1], "upper": 10.0 ** _bins[1:], "nodes": _counts}
        )
        _hist = (
            alt.Chart(_histogram)
            .mark_bar(color="#2563a6")
            .encode(
                x=alt.X(
                    "lower:Q",
                    scale=alt.Scale(type="log"),
                    title=f"{_method} (log scale)",
                ),
                x2="upper:Q",
                y=alt.Y("nodes:Q", title="Nodes"),
                tooltip=[
                    alt.Tooltip("lower:Q", format=".4g"),
                    alt.Tooltip("upper:Q", format=".4g"),
                    "nodes:Q",
                ],
            )
            .properties(
                width="container",
                height=centrality_top_count.value * 20,
                title="Distribution of positive scores",
            )
        )
        _distribution_outputs.append(
            mo.ui.altair_chart(_hist, chart_selection=False, legend_selection=False)
        )
    mo.vstack(
        [
            mo.md(f"### {_method}"),
            topic_widget_grid(
                [
                    mo.stat(f"{len(_values):,}", label="Nodes scored"),
                    mo.stat(f"{np.count_nonzero(_values == 0):,}", label="Zero scores"),
                    mo.stat(f"{_top_fraction:.1%}", label="Score share held by top 1%"),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                _bar, chart_selection=False, legend_selection=False
                            ),
                        ]
                    ),
                    mo.vstack(
                        _distribution_outputs
                        or [
                            mo.md(
                                "No positive scores are available for a logarithmic distribution."
                            ),
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.5,
            ),
            mo.md(
                "Zero scores are counted above and excluded from the logarithmic histogram. Top 1% uses ceil(0.01 × N) nodes. Equal scores are ordered by node ID for display; ranks used below average ties."
            ),
            mo.ui.table(
                _top.drop("account"),
                selection=None,
                pagination=True,
                show_data_types=False,
                show_column_summaries=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def control_pagerank_network():
    pagerank_network_neighbors = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        full_width=True,
        label="Top connected neighbors per PageRank node",
    )
    pagerank_network_labels = mo.ui.checkbox(value=False, label="Show all node IDs")
    mo.vstack(
        [
            mo.md("### Network of top PageRank nodes"),
            topic_widget_grid([pagerank_network_neighbors, pagerank_network_labels]),
            mo.md(
                "The **Top nodes** control above sets how many accounts are selected by PageRank. "
                "For each account, add the most-connected nodes it follows, ranked by total "
                "incoming + outgoing connections in this dataset. Ties use PageRank, then node ID. "
                "The plot includes all observed directed edges between the displayed nodes. "
                "This view always uses PageRank, regardless of the centrality measure selected above."
            ),
        ]
    )
    return pagerank_network_labels, pagerank_network_neighbors


@app.function(hide_code=True)
def build_pagerank_network_data(
    scores, adjacency, node_ids, top_count, neighbors_per_node
):
    """Select top PageRank nodes and each one's highest-degree outgoing neighbors."""
    ordered = scores.sort(["PageRank", "node"], descending=[True, False])
    top_nodes = ordered.head(top_count)["node"].to_list()
    top_set = set(top_nodes)
    metadata = {row["node"]: row for row in ordered.iter_rows(named=True)}
    ranks = {node: rank for rank, node in enumerate(ordered["node"], start=1)}
    selected = set(top_nodes)
    for node in top_nodes:
        index = int(np.searchsorted(node_ids, node))
        targets = node_ids[
            adjacency.indices[adjacency.indptr[index] : adjacency.indptr[index + 1]]
        ]
        neighbors = sorted(
            (int(target) for target in targets if target != node),
            key=lambda target: (
                -(metadata[target]["in_degree"] + metadata[target]["out_degree"]),
                -metadata[target]["PageRank"],
                target,
            ),
        )[:neighbors_per_node]
        selected.update(neighbors)
    selected_nodes = sorted(selected)
    indices = np.searchsorted(node_ids, selected_nodes)
    subgraph = adjacency[indices, :][:, indices].tocoo()
    edges = {
        (selected_nodes[int(source)], selected_nodes[int(target)])
        for source, target in zip(subgraph.row, subgraph.col)
        if source != target
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(selected_nodes)
    graph.add_edges_from(sorted(edges))
    positions = nx.spring_layout(graph.to_undirected(), seed=42, iterations=100)
    max_score = max(
        (metadata[node]["PageRank"] for node in selected_nodes), default=1.0
    )
    nodes = []
    for node in selected_nodes:
        row = metadata[node]
        nodes.append(
            {
                "node": str(node),
                "x": float(positions[node][0]),
                "y": float(positions[node][1]),
                "role": "Top PageRank node"
                if node in top_set
                else "Connected neighbor",
                "pagerank": row["PageRank"],
                "rank": ranks[node],
                "incoming": row["in_degree"],
                "outgoing": row["out_degree"],
                "connections": row["in_degree"] + row["out_degree"],
                "size": float(80 + 400 * np.sqrt(row["PageRank"] / max_score))
                if max_score
                else 80.0,
            }
        )
    links, arrows = [], []
    for source, target in sorted(edges):
        start, end = positions[source], positions[target]
        delta = end - start
        length = float(np.linalg.norm(delta))
        mutual = (target, source) in edges
        edge = {
            "source": str(source),
            "target": str(target),
            "relationship": "Mutual" if mutual else "One-way",
            "color": "#0d9488" if mutual else "#94a3b8",
            "x": float(start[0]),
            "y": float(start[1]),
            "x2": float(end[0]),
            "y2": float(end[1]),
        }
        if not mutual or source < target:
            links.append(edge)
        arrow_position = end - delta * min(0.045 / max(length, 1e-9), 0.35)
        arrows.append(
            {**edge, "x": float(arrow_position[0]), "y": float(arrow_position[1])}
        )
    return {
        "nodes": nodes,
        "links": links,
        "arrows": arrows,
        "top_nodes": len(top_nodes),
        "neighbors": len(selected - top_set),
        "directed_edges": len(edges),
    }


@app.cell(hide_code=True)
def compute_pagerank_network(
    centrality_scores,
    centrality_top_count,
    network_adjacency,
    network_node_ids,
    pagerank_network_neighbors,
):
    pagerank_network_data = build_pagerank_network_data(
        centrality_scores,
        network_adjacency,
        network_node_ids,
        centrality_top_count.value,
        pagerank_network_neighbors.value,
    )
    return (pagerank_network_data,)


@app.cell(hide_code=True)
def display_pagerank_network(pagerank_network_data, pagerank_network_labels):
    _network = pagerank_network_data
    mo.stop(
        not _network["nodes"], mo.md("No nodes are available for the PageRank network.")
    )
    _x = alt.X("x:Q", axis=None, scale=alt.Scale(zero=False, padding=35))
    _y = alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, padding=35))
    _edge_tooltip = [
        alt.Tooltip("source:N", title="From (follower)"),
        alt.Tooltip("target:N", title="To (followed account)"),
        alt.Tooltip("relationship:N", title="Relationship"),
    ]
    _links = (
        alt.Chart(alt.Data(values=_network["links"]))
        .mark_rule(
            opacity=0.35,
            strokeWidth=0.8,
        )
        .encode(
            x=_x,
            y=_y,
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=_edge_tooltip,
        )
    )
    _arrows = (
        alt.Chart(alt.Data(values=_network["arrows"]))
        .mark_point(
            shape="triangle-up",
            filled=True,
            size=28,
            opacity=0.7,
            angle=alt.ExprRef(
                expr="atan2(scale('x', datum.x2) - scale('x', datum.x), scale('y', datum.y) - scale('y', datum.y2)) * 180 / PI"
            ),
        )
        .encode(
            x=_x,
            y=_y,
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=_edge_tooltip,
        )
    )
    _node_base = alt.Chart(alt.Data(values=_network["nodes"]))
    _nodes = _node_base.mark_circle(stroke="white", strokeWidth=1.3, opacity=1).encode(
        x=_x,
        y=_y,
        size=alt.Size("size:Q", scale=None, legend=None),
        color=alt.Color(
            "role:N",
            title="Node group",
            scale=alt.Scale(
                domain=["Top PageRank node", "Connected neighbor"],
                range=["#f59e0b", "#3b82f6"],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("node:N", title="Node ID"),
            alt.Tooltip("role:N", title="Group"),
            alt.Tooltip("rank:Q", title="Global PageRank rank", format=",d"),
            alt.Tooltip("pagerank:Q", title="PageRank", format=".6g"),
            alt.Tooltip("incoming:Q", title="Incoming connections", format=",d"),
            alt.Tooltip("outgoing:Q", title="Outgoing connections", format=",d"),
            alt.Tooltip("connections:Q", title="Total connections", format=",d"),
        ],
    )
    _labels = (
        _node_base.transform_filter(
            "true"
            if pagerank_network_labels.value
            else alt.datum.role == "Top PageRank node"
        )
        .mark_text(dy=-17, fontSize=10, color="#475569")
        .encode(x=_x, y=_y, text="node:N")
    )
    pagerank_network_chart = (
        alt.layer(_links, _arrows, _nodes, _labels)
        .properties(
            width="container",
            height=580,
            title=f"Top {_network['top_nodes']} PageRank nodes and their connected neighbors",
        )
        .resolve_scale(color="independent")
        .configure_view(stroke=None)
        .interactive()
    )
    mo.vstack(
        [
            topic_widget_grid(
                [
                    mo.stat(f"{_network['top_nodes']:,}", label="Top PageRank nodes"),
                    mo.stat(
                        f"{_network['neighbors']:,}", label="Additional connected nodes"
                    ),
                    mo.stat(
                        f"{_network['directed_edges']:,}",
                        label="Displayed directed edges",
                    ),
                ]
            ),
            mo.ui.altair_chart(
                pagerank_network_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "Orange nodes are the top PageRank accounts; blue nodes are their selected outgoing neighbors. "
                "Node size represents PageRank. Arrows point from follower to followed account; teal edges are "
                "mutual and gray edges are one-way. Hover for node IDs, PageRank, and connection counts. "
                "Scroll to zoom, drag to pan, and double-click to reset. Shared neighbors appear once, "
                "and every ranked node is retained even if it has no outgoing connections. Self-loops are omitted."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def compare_centrality_rankings(
    centrality_diagnostics,
    centrality_methods,
    centrality_rank_agreement,
    centrality_scores,
):
    _agreement_chart = (
        alt.Chart(centrality_rank_agreement)
        .mark_rect()
        .encode(
            x=alt.X(
                "method:N",
                sort=centrality_methods,
                title=None,
                axis=alt.Axis(labelAngle=-30),
            ),
            y=alt.Y("comparison:N", sort=centrality_methods, title=None),
            color=alt.Color(
                "Spearman correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
            tooltip=[
                "method:N",
                "comparison:N",
                alt.Tooltip("Spearman correlation:Q", format=".3f"),
            ],
        )
        .properties(
            width="container", height=260, title="Agreement between centrality rankings"
        )
    )
    _pr_leader = centrality_scores.sort(
        ["PageRank", "node"], descending=[True, False]
    ).row(0, named=True)
    _authority_leader = centrality_scores.sort(
        ["HITS authority", "node"], descending=[True, False]
    ).row(0, named=True)
    _hub_leader = centrality_scores.sort(
        ["HITS hub", "node"], descending=[True, False]
    ).row(0, named=True)
    mo.vstack(
        [
            mo.md(
                f"### Comparing rankings\n\nNode **{_pr_leader['node']}** leads PageRank; node **{_authority_leader['node']}** leads HITS authority; node **{_hub_leader['node']}** leads HITS hub. The heatmap compares rankings over all nodes, including tied scores. Correlation near 1 means similar ordering; near 0 means little monotonic agreement."
            ),
            mo.ui.altair_chart(
                _agreement_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "PageRank spreads attention through the graph, while HITS emphasizes mutually reinforcing hubs and authorities. Disconnected regions can receive very different HITS weight. These measures do not assess shortest-path brokerage; betweenness and closeness require a separate treatment of reachability and computation cost on this graph."
            ),
            mo.accordion(
                {
                    "Convergence diagnostics": mo.ui.table(
                        centrality_diagnostics, selection=None, pagination=False
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def describe_components_and_reciprocity():
    mo.md(r"""
    ## 3. Components and reciprocity

    A **weak component** connects nodes when edge direction is ignored. In a
    **strong component**, every node can reach every other node along directed
    paths. A singleton strong component can still have incoming or outgoing edges;
    it is not necessarily isolated.

    All consolidated nodes are included, even nodes absent from the edge list.
    Self-loops are counted separately and excluded from the graph below, its
    density, and all reciprocity measures. Here, an **isolate** has no relationship
    to another node. Results describe this observed dataset.
    """)
    return


@app.cell(hide_code=True)
def import_sparse_graph_tools():
    import scipy.sparse as sparse
    from scipy.sparse.csgraph import connected_components

    return connected_components, sparse


@app.cell(hide_code=True)
def define_connectivity_analysis(connected_components):
    def analyze_connectivity(adjacency, node_ids: np.ndarray) -> dict:
        """Analyze a simple directed graph, keeping every node and excluding loops.

        Component IDs are the smallest original node ID in each component.
        Edge reciprocity counts both directions of each mutual pair; pair
        reciprocity counts that pair once. Undefined ratios remain null.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64)
        size = node_ids.size
        if adjacency.shape != (size, size) or np.unique(node_ids).size != size:
            raise ValueError("Adjacency shape must match unique node IDs")
        graph = adjacency.astype(bool).tocsr(copy=True)
        graph.eliminate_zeros()
        loops = int(np.count_nonzero(graph.diagonal()))
        graph.setdiag(False)
        graph.eliminate_zeros()
        memberships = {"node": node_ids}
        component_tables = []
        component_summaries = []
        for mode, label in [("weak", "Weak"), ("strong", "Strong")]:
            component_count, labels = connected_components(
                graph, directed=True, connection=mode, return_labels=True
            )
            sizes = np.bincount(labels, minlength=component_count)
            anchors = np.full(component_count, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(anchors, labels, node_ids)
            memberships[f"{mode}_component"] = anchors[labels]
            table = (
                pl.DataFrame(
                    {
                        "kind": [label] * component_count,
                        "component": anchors,
                        "nodes": sizes,
                        "node_share": sizes / size if size else np.array([]),
                    }
                )
                .sort(["nodes", "component"], descending=[True, False])
                .with_row_index("rank", offset=1)
            )
            component_tables.append(table)
            largest = int(sizes.max()) if component_count else 0
            component_summaries.append(
                {
                    "kind": label,
                    "components": component_count,
                    "largest_nodes": largest,
                    "largest_share": largest / size if size else None,
                    "singleton_components": int(np.count_nonzero(sizes == 1)),
                }
            )
        outgoing = np.diff(graph.indptr).astype(np.int64)
        incoming = np.bincount(graph.indices, minlength=size).astype(np.int64)
        mutual_graph = graph.multiply(graph.T).tocsr()
        mutual = np.diff(mutual_graph.indptr).astype(np.int64)
        edges = int(graph.nnz)
        reciprocal_edges = int(mutual_graph.nnz)
        mutual_pairs = reciprocal_edges // 2
        one_way_pairs = edges - reciprocal_edges
        connected_pairs = mutual_pairs + one_way_pairs
        outgoing_rate = np.divide(
            mutual, outgoing, out=np.full(size, np.nan), where=outgoing > 0
        )
        incoming_rate = np.divide(
            mutual, incoming, out=np.full(size, np.nan), where=incoming > 0
        )
        totals = {
            "nodes": size,
            "self_loops": loops,
            "edges": edges,
            "isolates": int(np.count_nonzero((incoming + outgoing) == 0)),
            "density": edges / (size * (size - 1)) if size > 1 else None,
            "reciprocal_edges": reciprocal_edges,
            "one_way_edges": one_way_pairs,
            "mutual_pairs": mutual_pairs,
            "one_way_pairs": one_way_pairs,
            "connected_pairs": connected_pairs,
            "edge_reciprocity": reciprocal_edges / edges if edges else None,
            "pair_reciprocity": mutual_pairs / connected_pairs
            if connected_pairs
            else None,
        }
        return {
            "adjacency": graph,
            "memberships": pl.DataFrame(memberships),
            "component_sizes": pl.concat(component_tables),
            "component_summary": pl.DataFrame(component_summaries),
            "totals": totals,
            "node_reciprocity": pl.DataFrame(
                {
                    "node": node_ids,
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "mutual_neighbors": mutual,
                    "outgoing_reciprocity": outgoing_rate,
                    "incoming_reciprocity": incoming_rate,
                }
            ).fill_nan(None),
        }

    return (analyze_connectivity,)


@app.cell(hide_code=True)
def compute_components_and_reciprocity(
    analyze_connectivity,
    edges_lf,
    nodes_lf,
    sparse,
):
    # A sparse matrix stores only observed edges. The sorted ID mapping preserves
    # feature-only nodes and makes component anchor IDs reproducible.
    network_node_ids = nodes_lf.select("node").sort("node").collect()["node"].to_numpy()
    mo.stop(
        network_node_ids.size == 0,
        mo.md("No nodes are available for connectivity analysis."),
    )
    _edges = edges_lf.select("from", "to").collect().to_numpy()
    _source_indices = np.searchsorted(network_node_ids, _edges[:, 0])
    _target_indices = np.searchsorted(network_node_ids, _edges[:, 1])
    if (
        np.any(_source_indices >= network_node_ids.size)
        or np.any(_target_indices >= network_node_ids.size)
        or not np.array_equal(network_node_ids[_source_indices], _edges[:, 0])
        or not np.array_equal(network_node_ids[_target_indices], _edges[:, 1])
    ):
        raise ValueError(
            "Every edge endpoint must be present in the consolidated node table"
        )
    _adjacency = sparse.csr_array(
        (np.ones(_edges.shape[0], dtype=bool), (_source_indices, _target_indices)),
        shape=(network_node_ids.size, network_node_ids.size),
    )
    connectivity_analysis = analyze_connectivity(_adjacency, network_node_ids)
    network_adjacency = connectivity_analysis["adjacency"]
    component_memberships = connectivity_analysis["memberships"]
    component_sizes = connectivity_analysis["component_sizes"]
    component_summary = connectivity_analysis["component_summary"]
    network_structure_totals = connectivity_analysis["totals"]
    node_reciprocity = connectivity_analysis["node_reciprocity"]

    reciprocity_comparison = pl.DataFrame(
        [
            {
                "basis": "Directed edges",
                "numerator": network_structure_totals["reciprocal_edges"],
                "denominator": network_structure_totals["edges"],
                "share": network_structure_totals["edge_reciprocity"],
            },
            {
                "basis": "Connected node pairs",
                "numerator": network_structure_totals["mutual_pairs"],
                "denominator": network_structure_totals["connected_pairs"],
                "share": network_structure_totals["pair_reciprocity"],
            },
        ]
    )
    _rate_summaries = []
    _histograms = []
    for _direction in ["outgoing", "incoming"]:
        _values = node_reciprocity[f"{_direction}_reciprocity"].drop_nulls().to_numpy()
        _counts, _bins = np.histogram(_values, bins=np.linspace(0, 1, 21))
        _histograms.append(
            pl.DataFrame(
                {
                    "direction": [_direction.title()] * _counts.size,
                    "lower": _bins[:-1],
                    "upper": _bins[1:],
                    "nodes": _counts,
                }
            )
        )
        _rate_summaries.append(
            {
                "direction": _direction.title(),
                "defined_nodes": _values.size,
                "undefined_nodes": network_node_ids.size - _values.size,
                "mean": float(_values.mean()) if _values.size else None,
                "median": float(np.median(_values)) if _values.size else None,
                "zero_rate_nodes": int(np.count_nonzero(_values == 0)),
                "fully_reciprocated_nodes": int(np.count_nonzero(_values == 1)),
            }
        )
    node_reciprocity_summary = pl.DataFrame(_rate_summaries)
    node_reciprocity_histogram = pl.concat(_histograms)
    return (
        component_memberships,
        component_sizes,
        component_summary,
        network_adjacency,
        network_node_ids,
        network_structure_totals,
        node_reciprocity,
        node_reciprocity_histogram,
        node_reciprocity_summary,
        reciprocity_comparison,
    )


@app.cell(hide_code=True)
def display_component_overview(component_summary, network_structure_totals):
    _weak = component_summary.filter(pl.col("kind") == "Weak").row(0, named=True)
    _strong = component_summary.filter(pl.col("kind") == "Strong").row(0, named=True)
    mo.vstack(
        [
            topic_widget_grid(
                [
                    mo.stat(f"{_weak['components']:,}", label="Weak components"),
                    mo.stat(f"{_strong['components']:,}", label="Strong components"),
                    mo.stat(
                        f"{_strong['largest_share']:.1%}",
                        label="Nodes in largest strong component",
                        caption=f"{_strong['largest_nodes']:,} of {network_structure_totals['nodes']:,} nodes",
                    ),
                    mo.stat(
                        f"{network_structure_totals['isolates']:,}",
                        label="Isolates",
                        caption="No non-self incoming or outgoing edges",
                    ),
                ]
            ),
            mo.ui.table(
                component_summary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={
                    "components": "{:,}",
                    "largest_nodes": "{:,}",
                    "largest_share": "{:.2%}",
                    "singleton_components": "{:,}",
                },
            ),
            mo.md(
                f"The graph has **{network_structure_totals['edges']:,} non-self directed edges** and **{network_structure_totals['self_loops']:,} self-loops**, which are excluded here. Directed density is **{network_structure_totals['density']:.6%}**, using n(n−1) possible non-self edges."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def select_component_connectivity():
    component_type_selector = mo.ui.dropdown(
        options=["Weak", "Strong"],
        value="Strong",
        label="Component definition",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("### Component controls"),
            topic_widget_grid([component_type_selector]),
        ]
    )
    return (component_type_selector,)


@app.cell(hide_code=True)
def display_component_size_distributions(
    component_memberships,
    component_sizes,
    component_type_selector,
):
    _kind = component_type_selector.value
    _components = component_sizes.filter(pl.col("kind") == _kind)
    _largest = _components.head(10)
    _frequency = _components.group_by("nodes").len(name="components").sort("nodes")
    component_rank_chart = (
        alt.Chart(_largest)
        .mark_circle(color="#2563a6", size=80)
        .encode(
            x=alt.X(
                "nodes:Q",
                title="Nodes per component (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", tickCount=5),
            ),
            y=alt.Y("rank:O", title="Component rank", sort="ascending"),
            tooltip=[
                alt.Tooltip("rank:O", title="Rank"),
                alt.Tooltip("component:Q", title="Smallest node ID", format="d"),
                alt.Tooltip("nodes:Q", title="Nodes", format=",d"),
                alt.Tooltip("node_share:Q", title="Share of all nodes", format=".3%"),
            ],
        )
        .properties(
            title=f"Largest {_kind.lower()} components (up to 10)",
            width="container",
            height=300,
        )
    )
    component_size_chart = (
        alt.Chart(_frequency)
        .mark_circle(color="#2563a6", size=75)
        .encode(
            x=alt.X(
                "nodes:Q",
                title="Nodes in a component (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", tickCount=5),
            ),
            y=alt.Y(
                "components:Q",
                title="Number of components (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", tickCount=5),
            ),
            tooltip=[
                alt.Tooltip("nodes:Q", title="Component size", format=",d"),
                alt.Tooltip(
                    "components:Q", title="Components of this size", format=",d"
                ),
            ],
        )
        .properties(
            title=f"Exact {_kind.lower()} component-size frequencies",
            width="container",
            height=300,
        )
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                component_rank_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                component_size_chart,
                                chart_selection=False,
                                legend_selection=False,
                            ),
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.5,
            ),
            mo.md(
                "Each dot represents an exact component size, not a histogram bin. A single component appears as one dot. Component IDs are their smallest original node ID; ranks sort by size, then ID."
            ),
            mo.accordion(
                {
                    "Component catalog": mo.ui.table(
                        _components,
                        selection=None,
                        pagination=True,
                        page_size=10,
                        show_data_types=False,
                        format_mapping={"node_share": "{:.3%}"},
                    ),
                    "Node-to-component memberships": mo.ui.table(
                        component_memberships,
                        selection=None,
                        pagination=True,
                        page_size=10,
                        show_data_types=False,
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def display_global_reciprocity(
    network_structure_totals,
    reciprocity_comparison,
):
    _totals = network_structure_totals
    _edge_rate = (
        f"{_totals['edge_reciprocity']:.2%}"
        if _totals["edge_reciprocity"] is not None
        else "Undefined"
    )
    _pair_rate = (
        f"{_totals['pair_reciprocity']:.2%}"
        if _totals["pair_reciprocity"] is not None
        else "Undefined"
    )
    _bars = (
        alt.Chart(reciprocity_comparison)
        .mark_bar(color="#2563a6")
        .encode(
            x=alt.X(
                "share:Q",
                title="Reciprocity within each denominator",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5),
            ),
            y=alt.Y(
                "basis:N", title=None, sort=["Directed edges", "Connected node pairs"]
            ),
            tooltip=[
                alt.Tooltip("basis:N", title="Denominator basis"),
                alt.Tooltip("numerator:Q", title="Reciprocated count", format=",d"),
                alt.Tooltip("denominator:Q", title="Total count", format=",d"),
                alt.Tooltip("share:Q", title="Reciprocity", format=".2%"),
            ],
        )
    )
    global_reciprocity_chart = (
        _bars
        + _bars.mark_text(align="left", dx=5, color="#333333").encode(
            text=alt.Text("share:Q", format=".1%")
        )
    ).properties(
        title="Reciprocity by counting unit",
        width="container",
        height=150,
    )
    mo.vstack(
        [
            mo.md("### Reciprocity across the network"),
            topic_widget_grid(
                [
                    mo.stat(
                        _edge_rate,
                        label="Directed-edge reciprocity",
                        caption=f"{_totals['reciprocal_edges']:,} / {_totals['edges']:,} directed edges",
                    ),
                    mo.stat(
                        _pair_rate,
                        label="Connected-pair reciprocity",
                        caption=f"{_totals['mutual_pairs']:,} / {_totals['connected_pairs']:,} connected pairs",
                    ),
                ]
            ),
            mo.md(
                f"There are **{_totals['mutual_pairs']:,} mutual pairs** and **{_totals['one_way_pairs']:,} one-way pairs**. If R is the mutual-pair count and U the one-way-pair count, directed-edge reciprocity is **2R / (2R + U)**, while connected-pair reciprocity is **R / (R + U)**. Pairs with no edge are outside both denominators. These rates are different views of the same relationships, not independent results."
            ),
            mo.ui.altair_chart(
                global_reciprocity_chart, chart_selection=False, legend_selection=False
            ),
            mo.accordion(
                {
                    "Exact numerators and denominators": mo.ui.table(
                        reciprocity_comparison,
                        selection=None,
                        pagination=False,
                        show_data_types=False,
                        show_column_summaries=False,
                        format_mapping={
                            "numerator": "{:,}",
                            "denominator": "{:,}",
                            "share": "{:.4%}",
                        },
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def display_node_reciprocity_distributions(
    node_reciprocity,
    node_reciprocity_histogram,
    node_reciprocity_summary,
):
    _tabs = {}
    node_reciprocity_charts = {}
    for _direction in ["Outgoing", "Incoming"]:
        _summary = node_reciprocity_summary.filter(
            pl.col("direction") == _direction
        ).row(0, named=True)
        _histogram = node_reciprocity_histogram.filter(
            pl.col("direction") == _direction
        )
        _chart = (
            alt.Chart(_histogram)
            .mark_bar(color="#2563a6", binSpacing=0)
            .encode(
                x=alt.X(
                    "lower:Q",
                    bin="binned",
                    title=f"{_direction} reciprocity per node",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", tickCount=5),
                ),
                x2="upper:Q",
                y=alt.Y(
                    "nodes:Q",
                    title="Nodes with a defined rate",
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("lower:Q", title="From (inclusive)", format=".0%"),
                    alt.Tooltip(
                        "upper:Q", title="To (exclusive, except 100%)", format=".0%"
                    ),
                    alt.Tooltip("nodes:Q", title="Nodes", format=",d"),
                ],
            )
            .properties(
                title=f"{_direction} reciprocity distribution",
                width="container",
                height=300,
            )
        )
        node_reciprocity_charts[_direction] = _chart
        _tabs[_direction] = mo.vstack(
            [
                mo.md(
                    f"**{_summary['defined_nodes']:,} nodes** have a defined rate; **{_summary['undefined_nodes']:,}** have no {_direction.lower()} edges and are excluded from this histogram, not assigned zero. Bins have width 5 percentage points; the final bin includes 100%."
                ),
                mo.ui.altair_chart(
                    _chart, chart_selection=False, legend_selection=False
                ),
            ]
        )
    mo.vstack(
        [
            mo.md("### Reciprocity for individual nodes"),
            mo.md(
                "**Outgoing reciprocity** is mutual neighbors / accounts followed: how many followed accounts follow back. **Incoming reciprocity** is mutual neighbors / followers: how many followers the account follows back. All counts here exclude self-loops. The mean below gives each eligible node equal weight; it is not global directed-edge reciprocity."
            ),
            mo.ui.table(
                node_reciprocity_summary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={
                    "mean": "{:.2%}",
                    "median": "{:.2%}",
                    "defined_nodes": "{:,}",
                    "undefined_nodes": "{:,}",
                    "zero_rate_nodes": "{:,}",
                    "fully_reciprocated_nodes": "{:,}",
                },
            ),
            mo.ui.tabs(_tabs),
            mo.accordion(
                {
                    "Per-node reciprocity data": mo.ui.table(
                        node_reciprocity,
                        selection=None,
                        pagination=True,
                        page_size=10,
                        show_data_types=False,
                        format_mapping={
                            "outgoing_reciprocity": "{:.2%}",
                            "incoming_reciprocity": "{:.2%}",
                        },
                    )
                }
            ),
            mo.md(
                "Methods: [SciPy connected components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html). The sparse adjacency, component memberships, and per-node reciprocity table are retained for subsequent notebook analyses."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def describe_combined_feature_pagerank():
    mo.md(r"""
    ## 4. Feature PageRank in the combined network

    This graph contains **account nodes** and **feature nodes**. It retains directed
    Twitter following edges and adds both account → feature and feature → account
    edges for every recorded feature membership. The two node types have separate
    index ranges, so equal numeric IDs never merge an account with a feature.

    At an account with both kinds of edges, the walk splits its probability between
    following edges and feature memberships. Within each kind, neighbors are equally
    likely. If only one kind exists, the walk uses it. A feature leads uniformly to
    accounts that have it. PageRank uses **85% continuation** and **15% restart**;
    restarts and dangling-node mass go uniformly to accounts.

    Compare the resulting feature ranking with simple frequency. Scores describe
    connectivity under this walk, not causal influence. Missing annotations are
    unobserved data, not evidence that an account lacks a feature.
    """)
    return


@app.cell(hide_code=True)
def control_combined_feature_pagerank():
    feature_pagerank_social_share = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.5,
        full_width=True,
        label="Share of account walk on following edges",
    )
    feature_pagerank_top_count = mo.ui.slider(
        start=10,
        stop=50,
        step=10,
        value=20,
        full_width=True,
        label="Top features to display",
    )
    feature_pagerank_min_accounts = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=1,
        full_width=True,
        label="Minimum accounts with feature (display only)",
    )
    mo.vstack(
        [
            mo.md("### Feature ranking controls"),
            topic_widget_grid(
                [
                    feature_pagerank_social_share,
                    feature_pagerank_top_count,
                    feature_pagerank_min_accounts,
                ]
            ),
            mo.md(
                "The walk-share control changes the model and recomputes PageRank. "
                "The other controls only change the displayed results. The complementary "
                "share goes to features; accounts with only one edge type use that type."
            ),
        ]
    )
    return (
        feature_pagerank_min_accounts,
        feature_pagerank_social_share,
        feature_pagerank_top_count,
    )


@app.cell(hide_code=True)
def define_combined_feature_pagerank(sparse):
    def prepare_feature_pagerank_walk(social_adjacency, incidence):
        """Cache row-normalized blocks for an account/account/feature random walk."""
        social = sparse.csr_array(social_adjacency, dtype=np.float64)
        memberships = sparse.csr_array(incidence, dtype=np.float64)
        if social.shape[0] == 0 or social.shape[0] != social.shape[1]:
            raise ValueError("The social graph must be nonempty and square")
        if memberships.shape[0] != social.shape[0]:
            raise ValueError("Membership rows must match the account graph")
        if np.any(social.data < 0) or np.any(memberships.data < 0):
            raise ValueError("Graph weights cannot be negative")
        outgoing = np.asarray(social.sum(axis=1)).ravel()
        account_features = np.asarray(memberships.sum(axis=1)).ravel()
        feature_accounts = np.asarray(memberships.sum(axis=0)).ravel()
        social_inverse = np.divide(
            1.0, outgoing, out=np.zeros_like(outgoing), where=outgoing > 0
        )
        account_inverse = np.divide(
            1.0,
            account_features,
            out=np.zeros_like(account_features),
            where=account_features > 0,
        )
        feature_inverse = np.divide(
            1.0,
            feature_accounts,
            out=np.zeros_like(feature_accounts),
            where=feature_accounts > 0,
        )
        return {
            "social_transpose": social.multiply(social_inverse[:, None]).T.tocsr(),
            "account_to_feature_transpose": memberships.multiply(
                account_inverse[:, None]
            ).T.tocsr(),
            "feature_to_account_transpose": memberships.T.multiply(
                feature_inverse[:, None]
            ).T.tocsr(),
            "has_social": outgoing > 0,
            "has_features": account_features > 0,
            "feature_dangling": feature_accounts == 0,
            "accounts": social.shape[0],
            "features": memberships.shape[1],
        }

    def compute_combined_feature_pagerank(
        walk, social_share=0.5, damping=0.85, tolerance=1e-10, max_iterations=500
    ):
        """Compute typed PageRank with uniform-account restarts and branch fallback."""
        if not 0 <= social_share <= 1 or not 0 < damping < 1:
            raise ValueError("Walk share must be in [0, 1] and damping in (0, 1)")
        social_weight = np.where(
            walk["has_social"] & walk["has_features"],
            social_share,
            walk["has_social"].astype(float),
        )
        feature_weight = np.where(walk["has_features"], 1.0 - social_weight, 0.0)
        dangling_accounts = ~(walk["has_social"] | walk["has_features"])
        accounts = np.full(walk["accounts"], 1.0 / walk["accounts"])
        features = np.zeros(walk["features"])

        def step(account_mass, feature_mass):
            dangling_mass = float(
                account_mass[dangling_accounts].sum()
                + feature_mass[walk["feature_dangling"]].sum()
            )
            next_accounts = (
                damping
                * (
                    walk["social_transpose"] @ (account_mass * social_weight)
                    + walk["feature_to_account_transpose"] @ feature_mass
                )
                + ((1.0 - damping) + damping * dangling_mass) / walk["accounts"]
            )
            next_features = damping * (
                walk["account_to_feature_transpose"] @ (account_mass * feature_weight)
            )
            return next_accounts, next_features

        for iteration in range(1, max_iterations + 1):
            next_accounts, next_features = step(accounts, features)
            change = float(
                np.abs(next_accounts - accounts).sum()
                + np.abs(next_features - features).sum()
            )
            accounts, features = next_accounts, next_features
            if change <= tolerance:
                break
        check_accounts, check_features = step(accounts, features)
        residual = float(
            np.abs(check_accounts - accounts).sum()
            + np.abs(check_features - features).sum()
        )
        diagnostics = {
            "iterations": iteration,
            "converged": residual <= tolerance,
            "l1_residual": residual,
            "total_mass": float(accounts.sum() + features.sum()),
            "account_mass": float(accounts.sum()),
            "feature_mass": float(features.sum()),
            "damping": damping,
            "social_share": social_share,
        }
        return accounts, features, diagnostics

    return compute_combined_feature_pagerank, prepare_feature_pagerank_walk


@app.cell(hide_code=True)
def build_combined_account_feature_graph(
    feats_lf,
    network_adjacency,
    network_node_ids,
    nodes_lf,
    prepare_feature_pagerank_walk,
    sparse,
):
    feature_pagerank_catalog = feats_lf.select("id", "name").sort("id").collect()
    _feature_ids = feature_pagerank_catalog["id"].to_numpy()
    _memberships = (
        nodes_lf.select("node", "features")
        .explode("features")
        .filter(pl.col("features").is_not_null())
        .unique(["node", "features"])
        .collect()
    )
    _account_indices = np.searchsorted(
        network_node_ids, _memberships["node"].to_numpy()
    )
    _feature_indices = np.searchsorted(
        _feature_ids, _memberships["features"].to_numpy()
    )
    if (
        np.any(_account_indices >= network_node_ids.size)
        or np.any(_feature_indices >= _feature_ids.size)
        or not np.array_equal(
            network_node_ids[_account_indices], _memberships["node"].to_numpy()
        )
        or not np.array_equal(
            _feature_ids[_feature_indices], _memberships["features"].to_numpy()
        )
    ):
        raise ValueError("Every membership must identify a known account and feature")
    feature_account_incidence = sparse.csr_array(
        (
            np.ones(_memberships.height, dtype=bool),
            (_account_indices, _feature_indices),
        ),
        shape=(network_node_ids.size, _feature_ids.size),
    )
    # Account indices come first; feature indices are offset by the account count.
    feature_pagerank_adjacency = sparse.block_array(
        [
            [network_adjacency, feature_account_incidence],
            [
                feature_account_incidence.T,
                sparse.csr_array((_feature_ids.size, _feature_ids.size), dtype=bool),
            ],
        ],
        format="csr",
    )
    feature_pagerank_vertices = pl.concat(
        [
            pl.DataFrame(
                {
                    "id": network_node_ids,
                    "node_type": ["account"] * network_node_ids.size,
                }
            ),
            feature_pagerank_catalog.select("id").with_columns(
                pl.lit("feature").alias("node_type")
            ),
        ]
    ).with_row_index("graph_index")
    feature_pagerank_walk = prepare_feature_pagerank_walk(
        network_adjacency, feature_account_incidence
    )
    feature_pagerank_account_counts = np.asarray(
        feature_account_incidence.sum(axis=0)
    ).ravel()
    feature_pagerank_graph_summary = {
        "accounts": network_node_ids.size,
        "features": _feature_ids.size,
        "memberships": int(feature_account_incidence.nnz),
        "social_edges": int(network_adjacency.nnz),
        "combined_edges": int(feature_pagerank_adjacency.nnz),
        "annotated_accounts": int(
            np.count_nonzero(feature_pagerank_walk["has_features"])
        ),
        "observed_features": int(np.count_nonzero(feature_pagerank_account_counts)),
    }
    return (
        feature_account_incidence,
        feature_pagerank_account_counts,
        feature_pagerank_catalog,
        feature_pagerank_graph_summary,
        feature_pagerank_walk,
    )


@app.cell(hide_code=True)
def compute_feature_pagerank_ranking(
    compute_combined_feature_pagerank,
    feature_pagerank_account_counts,
    feature_pagerank_catalog,
    feature_pagerank_graph_summary,
    feature_pagerank_social_share,
    feature_pagerank_walk,
):
    feature_pagerank_account_scores, _feature_scores, feature_pagerank_diagnostics = (
        compute_combined_feature_pagerank(
            feature_pagerank_walk,
            social_share=feature_pagerank_social_share.value,
        )
    )
    mo.stop(
        not feature_pagerank_diagnostics["converged"],
        mo.md(f"Feature PageRank did not converge: {feature_pagerank_diagnostics}"),
    )
    _feature_mass = feature_pagerank_diagnostics["feature_mass"]
    _membership_total = feature_pagerank_graph_summary["memberships"]
    feature_pagerank_scores = (
        feature_pagerank_catalog.with_columns(
            pl.Series("account_count", feature_pagerank_account_counts),
            pl.Series("pagerank", _feature_scores),
            pl.Series(
                "feature_share",
                _feature_scores / _feature_mass
                if _feature_mass > 0
                else np.zeros_like(_feature_scores),
            ),
        )
        .filter(pl.col("account_count") > 0)
        .with_columns(
            (
                pl.col("account_count") / feature_pagerank_graph_summary["accounts"]
            ).alias("account_coverage"),
            (pl.col("account_count") / _membership_total).alias("frequency_share"),
            pl.col("account_count")
            .rank(method="average", descending=True)
            .alias("frequency_rank"),
            pl.col("pagerank")
            .rank(method="average", descending=True)
            .alias("pagerank_rank"),
        )
        .with_columns(
            (pl.col("feature_share") / pl.col("frequency_share")).alias("share_lift"),
            (pl.col("frequency_rank") - pl.col("pagerank_rank")).alias("rank_gain"),
        )
        .sort(["pagerank", "id"], descending=[True, False])
    )
    _rank_a = feature_pagerank_scores["frequency_rank"].to_numpy()
    _rank_b = feature_pagerank_scores["pagerank_rank"].to_numpy()
    feature_pagerank_frequency_correlation = (
        float(np.corrcoef(_rank_a, _rank_b)[0, 1])
        if len(_rank_a) > 1 and np.std(_rank_a) > 0 and np.std(_rank_b) > 0
        else None
    )
    return (
        feature_pagerank_account_scores,
        feature_pagerank_diagnostics,
        feature_pagerank_frequency_correlation,
        feature_pagerank_scores,
    )


@app.cell(hide_code=True)
def display_feature_pagerank_analysis(
    feature_pagerank_diagnostics,
    feature_pagerank_frequency_correlation,
    feature_pagerank_graph_summary,
    feature_pagerank_min_accounts,
    feature_pagerank_scores,
    feature_pagerank_top_count,
):
    _summary = feature_pagerank_graph_summary
    _diagnostics = feature_pagerank_diagnostics
    _eligible = feature_pagerank_scores.filter(
        pl.col("account_count") >= feature_pagerank_min_accounts.value
    )
    _top = _eligible.head(feature_pagerank_top_count.value)
    _correlation = (
        f"{feature_pagerank_frequency_correlation:.3f}"
        if feature_pagerank_frequency_correlation is not None
        else "Undefined"
    )
    mo.output.append(
        mo.vstack(
            [
                topic_widget_grid(
                    [
                        mo.stat(
                            f"{_summary['accounts']:,}",
                            label="Account nodes",
                            caption=f"{_summary['annotated_accounts']:,} with recorded features",
                        ),
                        mo.stat(
                            f"{_summary['observed_features']:,}",
                            label="Observed feature nodes",
                            caption=f"{_summary['features']:,} labels in catalog",
                        ),
                        mo.stat(
                            f"{_summary['combined_edges']:,}",
                            label="Combined directed edges",
                            caption=f"{_summary['social_edges']:,} social + 2 × {_summary['memberships']:,} memberships",
                        ),
                        mo.stat(
                            _correlation,
                            label="Rank correlation with frequency",
                            caption="Spearman, all observed features; average ranks for ties",
                        ),
                    ]
                ),
                mo.md(
                    f"Converged in **{_diagnostics['iterations']} iterations**; L1 residual **{_diagnostics['l1_residual']:.2e}**. "
                    f"Accounts hold **{_diagnostics['account_mass']:.2%}** of PageRank mass and features **{_diagnostics['feature_mass']:.2%}**. "
                    "Feature share below is normalized across all feature nodes before display filters. "
                    "Unobserved catalog features have zero score and are excluded from rankings."
                ),
            ]
        )
    )
    mo.stop(
        _diagnostics["feature_mass"] <= 0,
        mo.md(
            "No PageRank mass reaches features at this walk share. Reduce the share on following edges."
        ),
    )
    mo.stop(
        _top.is_empty(),
        mo.md("No features meet the minimum account count. Lower the display filter."),
    )
    feature_pagerank_top_chart = (
        alt.Chart(_top)
        .mark_bar(color="#2563a6")
        .encode(
            x=alt.X(
                "feature_share:Q",
                title="Share of feature PageRank",
                axis=alt.Axis(format=".1%"),
            ),
            y=alt.Y(
                "name:N",
                title=None,
                sort=_top["name"].to_list(),
                axis=alt.Axis(labelLimit=180),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Feature"),
                alt.Tooltip("id:Q", title="Feature ID", format="d"),
                alt.Tooltip(
                    "feature_share:Q", title="Feature PageRank share", format=".4%"
                ),
                alt.Tooltip("account_count:Q", title="Accounts", format=",d"),
                alt.Tooltip(
                    "share_lift:Q", title="Share / frequency share", format=".2f"
                ),
                alt.Tooltip(
                    "rank_gain:Q", title="Rank gain over frequency", format=",.1f"
                ),
            ],
        )
        .properties(
            title="Top features by combined PageRank",
            width="container",
            height=max(300, _top.height * 22),
        )
    )
    _positive = _eligible.filter(pl.col("feature_share") > 0)
    _stride = max(1, int(np.ceil(_positive.height / 2500)))
    _sample = (
        _positive.sort("id")
        .with_row_index("sample_index")
        .filter(pl.col("sample_index") % _stride == 0)
        .drop("sample_index")
    )
    _sample = (
        pl.concat(
            [
                _sample,
                _top,
                _eligible.sort(["account_count", "id"], descending=[True, False]).head(
                    20
                ),
            ]
        )
        .unique("id")
        .filter(pl.col("feature_share") > 0)
    )
    _sample = _sample.with_columns(
        pl.when(pl.col("id").is_in(_top["id"].implode()))
        .then(pl.lit("Top features"))
        .otherwise(pl.lit("Other sampled features"))
        .alias("feature_group")
    )
    feature_pagerank_frequency_chart = (
        alt.Chart(_sample)
        .mark_circle(size=35, opacity=0.6)
        .encode(
            x=alt.X(
                "account_count:Q",
                title="Accounts with feature (log scale)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "feature_share:Q",
                title="Feature PageRank share (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format=".1e", tickCount=6),
            ),
            color=alt.Color(
                "feature_group:N",
                title="Feature group",
                scale=alt.Scale(
                    domain=["Other sampled features", "Top features"],
                    range=["#2563a6", "#f59e0b"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Feature"),
                alt.Tooltip("account_count:Q", title="Accounts", format=",d"),
                alt.Tooltip(
                    "feature_share:Q", title="Feature PageRank share", format=".4%"
                ),
                alt.Tooltip(
                    "share_lift:Q", title="Share / frequency share", format=".2f"
                ),
            ],
        )
        .properties(
            title="Frequency versus combined PageRank",
            width="container",
            height=max(300, _top.height * 22),
        )
        .interactive()
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                feature_pagerank_top_chart,
                                chart_selection=False,
                                legend_selection=False,
                            )
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                feature_pagerank_frequency_chart,
                                chart_selection=False,
                                legend_selection=False,
                            )
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.5,
            ),
            mo.md(
                f"Showing **{_top.height:,} of {_eligible.height:,} eligible features**. "
                f"The scatter displays **{_sample.height:,} features**, using a deterministic sample plus the highest PageRank and frequency features. "
                "The correlation uses every observed feature, regardless of display filters. "
                "A share lift above 1 means more feature PageRank share than its share of all recorded memberships; "
                "it is a descriptive comparison, not a significance test."
            ),
            mo.ui.table(
                _top.select(
                    "id",
                    "name",
                    "account_count",
                    "account_coverage",
                    "pagerank",
                    "feature_share",
                    "frequency_share",
                    "pagerank_rank",
                    "frequency_rank",
                    "rank_gain",
                    "share_lift",
                ),
                selection=None,
                pagination=True,
                page_size=10,
                show_data_types=False,
                show_column_summaries=False,
                format_mapping={
                    "account_coverage": "{:.2%}",
                    "pagerank": "{:.6g}",
                    "feature_share": "{:.4%}",
                    "frequency_share": "{:.4%}",
                    "share_lift": "{:.2f}",
                    "rank_gain": "{:+,.1f}",
                },
            ),
            mo.accordion(
                {
                    "All feature scores": mo.ui.table(
                        feature_pagerank_scores,
                        selection=None,
                        pagination=True,
                        page_size=10,
                        show_data_types=False,
                        show_column_summaries=False,
                        format_mapping={
                            "feature_share": "{:.4%}",
                            "share_lift": "{:.2f}",
                            "pagerank": "{:.6g}",
                        },
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def control_top_feature_network():
    feature_network_accounts_per_feature = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        full_width=True,
        label="Accounts per top feature",
    )
    feature_network_account_ranking = mo.ui.dropdown(
        options=["Combined PageRank", "Total connections"],
        value="Combined PageRank",
        full_width=True,
        label="Select connected accounts by",
    )
    feature_network_social_edges = mo.ui.checkbox(
        value=True, label="Show account following edges"
    )
    feature_network_account_labels = mo.ui.checkbox(
        value=False, label="Show account IDs on the plot"
    )
    mo.vstack(
        [
            mo.md("### Network of top features and accounts"),
            topic_widget_grid(
                [
                    feature_network_accounts_per_feature,
                    feature_network_account_ranking,
                    feature_network_social_edges,
                    feature_network_account_labels,
                ]
            ),
            mo.md(
                "Uses the **Top features to display** and **Minimum accounts with feature** controls above. "
                "For each top-ranked feature, select accounts that have it; shared accounts appear once. "
                "PageRank selection uses account scores from the combined graph. Connection-count selection "
                "uses incoming + outgoing social edges, with PageRank and node ID breaking ties. "
                "Scores are computed on the full combined graph, not recomputed on this displayed subset."
            ),
        ]
    )
    return (
        feature_network_account_labels,
        feature_network_account_ranking,
        feature_network_accounts_per_feature,
        feature_network_social_edges,
    )


@app.function(hide_code=True)
def build_top_feature_network_data(
    feature_scores,
    account_metadata,
    social_adjacency,
    incidence_columns,
    catalog_ids,
    top_count,
    accounts_per_feature,
    ranking="Combined PageRank",
    include_social=True,
):
    """Select top features and ranked member accounts, retaining all induced memberships."""
    top = feature_scores.sort(["pagerank", "id"], descending=[True, False]).head(
        top_count
    )
    if top.is_empty():
        return {
            "nodes": [],
            "links": [],
            "arrows": [],
            "features": 0,
            "accounts": 0,
            "memberships": 0,
            "social_edges": 0,
        }
    account_ids = account_metadata["node"].to_numpy()
    account_scores = account_metadata["combined_pagerank"].to_numpy()
    degree = (
        account_metadata["in_degree"] + account_metadata["out_degree"]
    ).to_numpy()
    feature_ids = top["id"].to_numpy()
    feature_columns = np.searchsorted(catalog_ids, feature_ids)
    if np.any(feature_columns >= len(catalog_ids)) or not np.array_equal(
        catalog_ids[feature_columns], feature_ids
    ):
        raise ValueError("Selected features must exist in the membership matrix")
    selected_accounts = set()
    for column in feature_columns:
        candidates = incidence_columns.indices[
            incidence_columns.indptr[column] : incidence_columns.indptr[column + 1]
        ]
        if ranking == "Combined PageRank":
            order = np.lexsort(
                (
                    account_ids[candidates],
                    -degree[candidates],
                    -account_scores[candidates],
                )
            )
        elif ranking == "Total connections":
            order = np.lexsort(
                (
                    account_ids[candidates],
                    -account_scores[candidates],
                    -degree[candidates],
                )
            )
        else:
            raise ValueError("Unknown account ranking")
        selected_accounts.update(
            int(index) for index in candidates[order[:accounts_per_feature]]
        )
    account_indices = np.array(sorted(selected_accounts), dtype=np.int64)
    feature_keys = [("feature", int(feature_id)) for feature_id in feature_ids]
    account_keys = [
        ("account", int(account_ids[index])) for index in account_indices
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(feature_keys + account_keys)
    memberships = incidence_columns[account_indices, :][:, feature_columns].tocoo()
    membership_edges = [
        (account_keys[int(row)], feature_keys[int(column)])
        for row, column in zip(memberships.row, memberships.col)
    ]
    for account_key, feature_key in membership_edges:
        graph.add_edge(account_key, feature_key)
        graph.add_edge(feature_key, account_key)
    social_edges = set()
    if include_social:
        social = social_adjacency[account_indices, :][:, account_indices].tocoo()
        social_edges = {
            (account_keys[int(source)], account_keys[int(target)])
            for source, target in zip(social.row, social.col)
            if source != target
        }
        graph.add_edges_from(sorted(social_edges))
    positions = nx.spring_layout(graph.to_undirected(), seed=42, iterations=100)
    max_score = max(
        float(top["pagerank"].max()),
        float(account_scores[account_indices].max())
        if account_indices.size
        else 0.0,
    )
    nodes = []
    for row in top.iter_rows(named=True):
        key = ("feature", int(row["id"]))
        label = row["name"] if len(row["name"]) <= 28 else row["name"][:27] + "…"
        nodes.append(
            {
                "key": f"feature:{row['id']}",
                "id": str(row["id"]),
                "label": label,
                "name": row["name"],
                "kind": "Feature",
                "x": float(positions[key][0]),
                "y": float(positions[key][1]),
                "pagerank": row["pagerank"],
                "details": f"{row['account_count']:,} accounts; {row['feature_share']:.4%} of feature PageRank",
                "size": float(90 + 400 * np.sqrt(row["pagerank"] / max_score))
                if max_score
                else 90.0,
            }
        )
    for index, key in zip(account_indices, account_keys):
        row = account_metadata.row(int(index), named=True)
        nodes.append(
            {
                "key": f"account:{row['node']}",
                "id": str(row["node"]),
                "label": str(row["node"]),
                "name": str(row["node"]),
                "kind": "Account",
                "x": float(positions[key][0]),
                "y": float(positions[key][1]),
                "pagerank": row["combined_pagerank"],
                "details": f"Incoming: {row['in_degree']:,}; outgoing: {row['out_degree']:,}",
                "size": float(
                    90 + 400 * np.sqrt(row["combined_pagerank"] / max_score)
                )
                if max_score
                else 90.0,
            }
        )
    names = {(row["kind"].lower(), int(row["id"])): row["name"] for row in nodes}
    links, arrows = [], []
    for account_key, feature_key in membership_edges:
        start, end = positions[account_key], positions[feature_key]
        links.append(
            {
                "source": f"account:{account_key[1]}",
                "target": f"feature:{feature_key[1]}",
                "source_label": names[account_key],
                "target_label": names[feature_key],
                "edge_type": "Account–feature membership",
                "direction": "Both directions",
                "x": float(start[0]),
                "y": float(start[1]),
                "x2": float(end[0]),
                "y2": float(end[1]),
            }
        )
    for source, target in sorted(social_edges):
        start, end = positions[source], positions[target]
        delta = end - start
        length = float(np.linalg.norm(delta))
        mutual = (target, source) in social_edges
        edge = {
            "source": f"account:{source[1]}",
            "target": f"account:{target[1]}",
            "source_label": names[source],
            "target_label": names[target],
            "edge_type": "Account follows account",
            "direction": "Mutual" if mutual else "One-way",
            "x": float(start[0]),
            "y": float(start[1]),
            "x2": float(end[0]),
            "y2": float(end[1]),
        }
        if not mutual or source < target:
            links.append(edge)
        point = end - delta * min(0.045 / max(length, 1e-9), 0.35)
        arrows.append({**edge, "x": float(point[0]), "y": float(point[1])})
    return {
        "nodes": nodes,
        "links": links,
        "arrows": arrows,
        "features": top.height,
        "accounts": len(account_indices),
        "memberships": len(membership_edges),
        "social_edges": len(social_edges),
    }


@app.cell(hide_code=True)
def index_feature_network_memberships(feature_account_incidence):
    feature_network_incidence_columns = feature_account_incidence.tocsc()
    return (feature_network_incidence_columns,)


@app.cell(hide_code=True)
def compute_top_feature_network(
    centrality_scores,
    feature_network_account_ranking,
    feature_network_accounts_per_feature,
    feature_network_incidence_columns,
    feature_network_social_edges,
    feature_pagerank_account_scores,
    feature_pagerank_catalog,
    feature_pagerank_min_accounts,
    feature_pagerank_scores,
    feature_pagerank_top_count,
    network_adjacency,
    network_node_ids,
):
    _accounts = centrality_scores.select("node", "in_degree", "out_degree").sort("node")
    if not np.array_equal(_accounts["node"].to_numpy(), network_node_ids):
        raise ValueError("Account score rows must align with the graph's account IDs")
    _accounts = _accounts.with_columns(
        pl.Series("combined_pagerank", feature_pagerank_account_scores)
    )
    feature_network_plot_data = build_top_feature_network_data(
        feature_pagerank_scores.filter(
            pl.col("account_count") >= feature_pagerank_min_accounts.value
        ),
        _accounts,
        network_adjacency,
        feature_network_incidence_columns,
        feature_pagerank_catalog["id"].to_numpy(),
        feature_pagerank_top_count.value,
        feature_network_accounts_per_feature.value,
        feature_network_account_ranking.value,
        feature_network_social_edges.value,
    )
    return (feature_network_plot_data,)


@app.cell(hide_code=True)
def display_top_feature_network(
    feature_network_account_labels,
    feature_network_plot_data,
    feature_pagerank_diagnostics,
):
    _network = feature_network_plot_data
    mo.stop(
        feature_pagerank_diagnostics["feature_mass"] <= 0,
        mo.md(
            "No PageRank mass reaches features. Reduce the following-edge walk share above."
        ),
    )
    mo.stop(
        not _network["nodes"],
        mo.md(
            "No features meet the selected minimum account count. Lower the display filter above."
        ),
    )
    _x = alt.X("x:Q", axis=None, scale=alt.Scale(zero=False, padding=40))
    _y = alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, padding=40))
    _links = (
        alt.Chart(alt.Data(values=_network["links"]))
        .mark_rule(opacity=0.4, strokeWidth=1)
        .encode(
            x=_x,
            y=_y,
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color(
                "edge_type:N",
                title="Connection type",
                scale=alt.Scale(
                    domain=["Account–feature membership", "Account follows account"],
                    range=["#8b5cf6", "#94a3b8"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("source_label:N", title="From / account"),
                alt.Tooltip("target_label:N", title="To / feature"),
                alt.Tooltip("edge_type:N", title="Connection"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
    )
    _arrows = (
        alt.Chart(alt.Data(values=_network["arrows"]))
        .mark_point(
            shape="triangle-up",
            color="#94a3b8",
            filled=True,
            size=28,
            opacity=0.75,
            angle=alt.ExprRef(
                expr="atan2(scale('x', datum.x2) - scale('x', datum.x), scale('y', datum.y) - scale('y', datum.y2)) * 180 / PI"
            ),
        )
        .encode(
            x=_x,
            y=_y,
            tooltip=[
                alt.Tooltip("source_label:N", title="Follower"),
                alt.Tooltip("target_label:N", title="Followed account"),
            ],
        )
    )
    _node_base = alt.Chart(alt.Data(values=_network["nodes"]))
    _nodes = _node_base.mark_point(
        filled=True, stroke="white", strokeWidth=1.3, opacity=1
    ).encode(
        x=_x,
        y=_y,
        size=alt.Size("size:Q", scale=None, legend=None),
        color=alt.Color(
            "kind:N",
            title="Node type",
            scale=alt.Scale(
                domain=["Feature", "Account"], range=["#f59e0b", "#3b82f6"]
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        shape=alt.Shape(
            "kind:N",
            title="Node type",
            scale=alt.Scale(domain=["Feature", "Account"], range=["diamond", "circle"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("kind:N", title="Node type"),
            alt.Tooltip("name:N", title="Feature / account"),
            alt.Tooltip("id:N", title="Original ID"),
            alt.Tooltip("pagerank:Q", title="Combined PageRank", format=".6g"),
            alt.Tooltip("details:N", title="Details"),
        ],
    )
    _labels = (
        _node_base.transform_filter(
            "true"
            if feature_network_account_labels.value
            else alt.datum.kind == "Feature"
        )
        .mark_text(dy=-17, fontSize=10, color="#475569")
        .encode(x=_x, y=_y, text="label:N")
    )
    feature_network_chart = (
        alt.layer(_links, _arrows, _nodes, _labels)
        .properties(
            width="container",
            height=620,
            title=f"Top {_network['features']} features and their connected accounts",
        )
        .resolve_scale(color="independent")
        .configure_view(stroke=None)
        .interactive()
    )
    mo.vstack(
        [
            topic_widget_grid(
                [
                    mo.stat(f"{_network['features']:,}", label="Top features"),
                    mo.stat(f"{_network['accounts']:,}", label="Connected accounts"),
                    mo.stat(
                        f"{_network['memberships']:,}",
                        label="Displayed memberships",
                        caption="Each represents two directed graph edges",
                    ),
                    mo.stat(
                        f"{_network['social_edges']:,}",
                        label="Displayed following edges",
                    ),
                ]
            ),
            mo.ui.altair_chart(
                feature_network_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "**Orange diamonds are features; blue circles are accounts.** Purple lines represent "
                "bidirectional account–feature memberships. Gray arrows show following edges from follower "
                "to followed account. All memberships between displayed features and accounts are included, "
                "even if an account was selected for another feature. Node size uses the same combined "
                "PageRank scale for both node types. Hover for full names, IDs, scores, and counts. "
                "Scroll to zoom, drag to pan, and double-click to reset. Feature and account IDs occupy "
                "separate namespaces. This is a selected view of the graph; hidden accounts and features "
                "still contribute to the PageRank scores."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def control_combined_account_ranking():
    combined_account_top_count = mo.ui.slider(
        start=10,
        stop=50,
        step=10,
        value=20,
        full_width=True,
        label="Top accounts from the combined graph",
    )
    mo.vstack(
        [
            mo.md("### Account PageRank after including features"),
            topic_widget_grid([combined_account_top_count]),
            mo.md(
                "These account scores come from the same full account-and-feature graph used above. "
                "The following-edge walk share above controls both account and feature rankings. "
                "Account share is the account's raw combined PageRank divided by the total mass on "
                "account nodes. This normalization leaves account ordering unchanged and makes shares "
                "comparable with the account-only model. A higher score is importance under this "
                "random walk, not a causal effect of having features."
            ),
        ]
    )
    return (combined_account_top_count,)


@app.cell(hide_code=True)
def rank_accounts_in_combined_graph(
    centrality_scores,
    feature_account_incidence,
    feature_pagerank_account_scores,
    feature_pagerank_diagnostics,
    network_node_ids,
):
    _account_mass = feature_pagerank_diagnostics["account_mass"]
    mo.stop(
        not feature_pagerank_diagnostics["converged"] or _account_mass <= 0,
        mo.md(
            "The combined PageRank model must converge with positive account mass before ranking accounts."
        ),
    )
    _combined = pl.DataFrame(
        {
            "node": network_node_ids,
            "combined_pagerank": feature_pagerank_account_scores,
            "account_share": feature_pagerank_account_scores / _account_mass,
            "recorded_features": np.asarray(
                feature_account_incidence.sum(axis=1)
            ).ravel(),
        }
    )
    combined_account_ranking = (
        centrality_scores.select(
            "node",
            "in_degree",
            "out_degree",
            pl.col("PageRank").alias("account_only_pagerank"),
        )
        .join(_combined, on="node", how="inner", validate="1:1")
        .with_columns(
            pl.col("combined_pagerank")
            .rank(method="average", descending=True)
            .alias("combined_rank"),
            pl.col("account_only_pagerank")
            .rank(method="average", descending=True)
            .alias("account_only_rank"),
        )
        .with_columns(
            (pl.col("account_only_rank") - pl.col("combined_rank")).alias("rank_gain")
        )
        .sort(["combined_pagerank", "node"], descending=[True, False])
    )
    _baseline_ranks = combined_account_ranking["account_only_rank"].to_numpy()
    _combined_ranks = combined_account_ranking["combined_rank"].to_numpy()
    combined_account_rank_correlation = (
        float(np.corrcoef(_baseline_ranks, _combined_ranks)[0, 1])
        if len(_baseline_ranks) > 1
        and np.std(_baseline_ranks) > 0
        and np.std(_combined_ranks) > 0
        else None
    )
    return combined_account_rank_correlation, combined_account_ranking


@app.cell(hide_code=True)
def display_combined_account_ranking(
    combined_account_rank_correlation,
    combined_account_ranking,
    combined_account_top_count,
    feature_pagerank_diagnostics,
):
    _top = combined_account_ranking.head(combined_account_top_count.value).with_columns(
        pl.col("node").cast(pl.String).alias("account")
    )
    _correlation = (
        f"{combined_account_rank_correlation:.3f}"
        if combined_account_rank_correlation is not None
        else "Undefined"
    )
    _top_ids = _top["node"]
    _baseline_top = combined_account_ranking.sort(
        ["account_only_pagerank", "node"], descending=[True, False]
    ).head(combined_account_top_count.value)
    _overlap = len(set(_top_ids.to_list()) & set(_baseline_top["node"].to_list()))
    combined_account_top_chart = (
        alt.Chart(_top)
        .mark_bar(color="#2563a6")
        .encode(
            x=alt.X(
                "account_share:Q",
                title="Share of account PageRank in combined graph",
                axis=alt.Axis(format=".2%"),
            ),
            y=alt.Y("account:N", title="Account ID", sort=_top["account"].to_list()),
            tooltip=[
                alt.Tooltip("account:N", title="Account ID"),
                alt.Tooltip(
                    "account_share:Q", title="Combined account share", format=".4%"
                ),
                alt.Tooltip(
                    "combined_pagerank:Q", title="Raw combined PageRank", format=".6g"
                ),
                alt.Tooltip("combined_rank:Q", title="Combined rank", format=",.1f"),
                alt.Tooltip(
                    "account_only_rank:Q", title="Account-only rank", format=",.1f"
                ),
                alt.Tooltip("rank_gain:Q", title="Rank gain", format="+,.1f"),
                alt.Tooltip(
                    "recorded_features:Q", title="Recorded features", format=",d"
                ),
                alt.Tooltip("in_degree:Q", title="Incoming connections", format=",d"),
                alt.Tooltip("out_degree:Q", title="Outgoing connections", format=",d"),
            ],
        )
        .properties(
            title="Top accounts in account-and-feature graph",
            width="container",
            height=max(300, _top.height * 22),
        )
    )
    _stride = max(1, int(np.ceil(combined_account_ranking.height / 2500)))
    _sample = (
        combined_account_ranking.sort("node")
        .with_row_index("sample_index")
        .filter(pl.col("sample_index") % _stride == 0)
        .drop("sample_index")
    )
    _sample = (
        pl.concat(
            [
                _sample,
                _top.drop("account"),
                _baseline_top,
            ]
        )
        .unique("node")
        .with_columns(
            pl.col("node").cast(pl.String).alias("account"),
            pl.when(pl.col("node").is_in(_top_ids.implode()))
            .then(pl.lit("Top combined accounts"))
            .otherwise(pl.lit("Other sampled accounts"))
            .alias("group"),
        )
    )
    _rank_domain = [1, combined_account_ranking.height]
    _rank_x = alt.X(
        "account_only_rank:Q",
        title="Account-only PageRank rank (log scale)",
        scale=alt.Scale(type="log", domain=_rank_domain),
        axis=alt.Axis(format="~s", tickCount=5),
    )
    _rank_y = alt.Y(
        "combined_rank:Q",
        title="Combined PageRank rank (log scale)",
        scale=alt.Scale(type="log", domain=_rank_domain, reverse=True),
        axis=alt.Axis(format="~s", tickCount=5),
    )
    _equality = (
        alt.Chart(
            pl.DataFrame(
                {"account_only_rank": _rank_domain, "combined_rank": _rank_domain}
            )
        )
        .mark_line(
            color="#64748b",
            strokeDash=[5, 5],
            opacity=0.7,
        )
        .encode(x=_rank_x, y=_rank_y)
    )
    _scatter = (
        alt.Chart(_sample)
        .mark_circle(size=35, opacity=0.6)
        .encode(
            x=_rank_x,
            y=_rank_y,
            color=alt.Color(
                "group:N",
                title="Account group",
                scale=alt.Scale(
                    domain=["Other sampled accounts", "Top combined accounts"],
                    range=["#2563a6", "#f59e0b"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("account:N", title="Account ID"),
                alt.Tooltip("combined_rank:Q", title="Combined rank", format=",.1f"),
                alt.Tooltip(
                    "account_only_rank:Q", title="Account-only rank", format=",.1f"
                ),
                alt.Tooltip("rank_gain:Q", title="Rank gain", format="+,.1f"),
                alt.Tooltip(
                    "recorded_features:Q", title="Recorded features", format=",d"
                ),
            ],
        )
    )
    combined_account_comparison_chart = (
        (_equality + _scatter)
        .properties(
            width="container",
            height=max(300, _top.height * 22),
            title="Account rank changes after including features",
        )
        .interactive()
    )
    mo.vstack(
        [
            topic_widget_grid(
                [
                    mo.stat(
                        f"{combined_account_ranking.height:,}", label="Accounts ranked"
                    ),
                    mo.stat(
                        f"{feature_pagerank_diagnostics['account_mass']:.2%}",
                        label="Combined PageRank mass on accounts",
                    ),
                    mo.stat(
                        _correlation,
                        label="Rank correlation with account-only model",
                        caption="Spearman, all accounts; average ranks for ties",
                    ),
                    mo.stat(
                        f"{_overlap} / {_top.height}",
                        label="Top accounts shared with account-only model",
                    ),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                combined_account_top_chart,
                                chart_selection=False,
                                legend_selection=False,
                            )
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.ui.altair_chart(
                                combined_account_comparison_chart,
                                chart_selection=False,
                                legend_selection=False,
                            )
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.5,
            ),
            mo.md(
                f"The rank comparison shows **{_sample.height:,} accounts**, using a deterministic sample plus the top accounts from both models. "
                "Lower rank numbers are better. Points above the dashed equality line have improved ranks in the combined model. "
                "The correlation and ranking table use all accounts. Positive rank gain means an improved rank. "
                "Both models retain accounts without recorded features; adding features can change their scores too. "
                "Ranking ties use average ranks, with account IDs ordering the displayed rows."
            ),
            mo.ui.table(
                _top.drop("account"),
                selection=None,
                pagination=True,
                page_size=10,
                show_data_types=False,
                show_column_summaries=False,
                format_mapping={
                    "combined_pagerank": "{:.6g}",
                    "account_share": "{:.4%}",
                    "account_only_pagerank": "{:.6g}",
                    "rank_gain": "{:+,.1f}",
                    "combined_rank": "{:,.1f}",
                    "account_only_rank": "{:,.1f}",
                },
            ),
            mo.accordion(
                {
                    "All account scores from the combined graph": mo.ui.table(
                        combined_account_ranking,
                        selection=None,
                        pagination=True,
                        page_size=10,
                        show_data_types=False,
                        show_column_summaries=False,
                        format_mapping={
                            "combined_pagerank": "{:.6g}",
                            "account_share": "{:.4%}",
                            "rank_gain": "{:+,.1f}",
                        },
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def describe_ranked_profile_clustering():
    mo.md(r"""
    ## 5. Cluster top-ranked accounts and features

    **Do highly ranked accounts share annotations? Do highly ranked features reach
    the same accounts?** Compare K-means and DBSCAN separately for each population.
    PageRank selects the candidates; scores and ranks are not clustering inputs.

    - **Accounts:** binary profiles over every recorded feature.
    - **Features:** binary profiles over every account in the dataset, including
      accounts outside the top-ranked selection.

    Profiles are L2-normalized so overlap matters more than profile size. K-means
    minimizes squared Euclidean distance to centroids and assigns every usable profile;
    DBSCAN groups densely connected profiles within radius ε and can label others
    as noise. The same normalized profiles are used for both algorithms. On these
    profiles, Euclidean distance is √(2 × cosine distance); disjoint profiles are
    √2 ≈ 1.414 apart. Sparse annotations can therefore make DBSCAN highly sensitive
    to ε near that value.

    Accounts without recorded annotations are **unclusterable**, rather than noise.
    Missing annotations remain unobserved data. These are annotation-similarity
    groups within a rank-selected sample, not detected communities of following edges.
    """)
    return


@app.cell(hide_code=True)
def control_ranked_profile_clustering():
    ranked_cluster_top_count = mo.ui.slider(
        start=20,
        stop=500,
        step=20,
        value=200,
        label="Top candidates per population",
        full_width=True,
    )
    ranked_cluster_account_ranking = mo.ui.dropdown(
        options=["Combined PageRank", "Account-only PageRank"],
        value="Combined PageRank",
        allow_select_none=False,
        label="Account candidate ranking",
        full_width=True,
    )
    ranked_cluster_k = mo.ui.slider(
        start=2,
        stop=15,
        value=6,
        step=1,
        label="K-means: requested clusters (k)",
        full_width=True,
    )
    ranked_cluster_account_eps = mo.ui.slider(
        start=0.05,
        stop=1.45,
        step=0.01,
        value=1.38,
        label="Account DBSCAN: radius ε",
        full_width=True,
    )
    ranked_cluster_feature_eps = mo.ui.slider(
        start=0.05,
        stop=1.45,
        step=0.01,
        value=1.30,
        label="Feature DBSCAN: radius ε",
        full_width=True,
    )
    ranked_cluster_min_samples = mo.ui.slider(
        start=2,
        stop=20,
        step=1,
        value=4,
        label="DBSCAN: minimum neighbors (includes self)",
        full_width=True,
    )
    mo.vstack(
        [
            topic_widget_grid(
                [
                    ranked_cluster_top_count,
                    ranked_cluster_account_ranking,
                    ranked_cluster_k,
                    ranked_cluster_account_eps,
                    ranked_cluster_feature_eps,
                    ranked_cluster_min_samples,
                ]
            ),
            mo.md(
                "Features use combined-network PageRank and the minimum-account filter above; "
                "accounts use the selected ranking. Equal scores are ordered by numeric ID. "
                "Candidate limits are independent of the ranking display limits above. "
                "K-means uses seed 42 and 10 initializations. Cluster numbers are local to each fit."
            ),
        ]
    )
    return (
        ranked_cluster_account_eps,
        ranked_cluster_account_ranking,
        ranked_cluster_feature_eps,
        ranked_cluster_k,
        ranked_cluster_min_samples,
        ranked_cluster_top_count,
    )


@app.cell(hide_code=True)
def define_ranked_cluster_profiles(sparse):
    def prepare_ranked_cluster_profiles(metadata, incidence):
        """Retain rank order, exclude empty profiles, and prepare a bounded comparison."""
        binary = sparse.csr_matrix(incidence, dtype=np.float64)
        binary.eliminate_zeros()
        binary.sort_indices()
        if binary.shape[0] != metadata.height:
            raise ValueError("Profile rows must match the ranked candidate table")
        if binary.nnz and (
            not np.isfinite(binary.data).all() or np.any(binary.data != 1)
        ):
            raise ValueError("Clustering requires finite binary memberships")
        observed = np.diff(binary.indptr) > 0
        active_columns = np.flatnonzero(np.asarray(binary.sum(axis=0)).ravel())
        usable = binary[observed][:, active_columns]
        profiles = normalize(usable) if usable.shape[0] else usable
        # Duplicate binary supports also identify duplicate normalized profiles.
        unique_count = len(
            {
                tuple(usable.indices[usable.indptr[row] : usable.indptr[row + 1]])
                for row in range(usable.shape[0])
            }
        )
        coordinates = np.zeros((usable.shape[0], 2))
        if usable.shape[0]:
            with threadpool_limits(limits=2):
                distances = pairwise_distances(profiles, metric="euclidean")
                if unique_count > 1 and min(profiles.shape) > 1:
                    coordinates = TruncatedSVD(
                        n_components=2, random_state=42
                    ).fit_transform(profiles)
                else:
                    coordinates[:, 0] = 1.0
            np.fill_diagonal(distances, 0.0)
            # sklearn accepts precomputed DBSCAN distances only when nonnegative.
            distances = np.maximum(distances, 0.0)
        else:
            distances = np.empty((0, 0))
        return {
            "metadata": metadata,
            "observed": observed,
            "binary": usable,
            "profiles": profiles,
            "active_columns": active_columns,
            "distances": distances,
            "coordinates": coordinates,
            "unique_count": unique_count,
        }

    return (prepare_ranked_cluster_profiles,)


@app.cell(hide_code=True)
def prepare_ranked_cluster_populations(
    combined_account_ranking,
    feature_account_incidence,
    feature_network_incidence_columns,
    feature_pagerank_catalog,
    feature_pagerank_min_accounts,
    feature_pagerank_scores,
    network_node_ids,
    prepare_ranked_cluster_profiles,
    ranked_cluster_account_ranking,
    ranked_cluster_top_count,
):
    _account_score = (
        "combined_pagerank"
        if ranked_cluster_account_ranking.value == "Combined PageRank"
        else "account_only_pagerank"
    )
    _account_rank = (
        "combined_rank"
        if ranked_cluster_account_ranking.value == "Combined PageRank"
        else "account_only_rank"
    )
    _accounts = (
        combined_account_ranking.sort(
            [_account_score, "node"], descending=[True, False]
        )
        .head(ranked_cluster_top_count.value)
        .select(
            pl.col("node").alias("id"),
            pl.col("node").cast(pl.String).alias("label"),
            pl.col(_account_rank).alias("rank"),
            pl.col(_account_score).alias("pagerank"),
            pl.col("recorded_features").alias("memberships"),
        )
    )
    _features = (
        feature_pagerank_scores.filter(
            pl.col("account_count") >= feature_pagerank_min_accounts.value
        )
        .sort(["pagerank", "id"], descending=[True, False])
        .head(ranked_cluster_top_count.value)
        .select(
            "id",
            pl.col("name").alias("label"),
            pl.col("pagerank_rank").alias("rank"),
            "pagerank",
            pl.col("account_count").alias("memberships"),
        )
    )
    _account_positions = np.searchsorted(network_node_ids, _accounts["id"].to_numpy())
    _feature_positions = np.searchsorted(
        feature_pagerank_catalog["id"].to_numpy(), _features["id"].to_numpy()
    )
    ranked_cluster_profiles = {
        "Accounts": prepare_ranked_cluster_profiles(
            _accounts, feature_account_incidence[_account_positions, :]
        ),
        "Features": prepare_ranked_cluster_profiles(
            _features, feature_network_incidence_columns[:, _feature_positions].T
        ),
    }
    return (ranked_cluster_profiles,)


@app.function(hide_code=True)
def fit_ranked_profile_clusters(
    prepared, requested_k, eps, min_samples, coordinate_labels
):
    """Fit in full profile space; return labels, diagnostics, and interpretable groups."""
    metadata = prepared["metadata"]
    observed = prepared["observed"]
    profiles = prepared["profiles"]
    size = profiles.shape[0]
    effective_k = min(requested_k, prepared["unique_count"], size)
    with threadpool_limits(limits=2):
        kmeans_labels = (
            KMeans(n_clusters=effective_k, random_state=42, n_init=10).fit_predict(
                profiles
            )
            if size
            else np.array([], dtype=int)
        )
        dbscan_labels = (
            DBSCAN(
                eps=eps, min_samples=min_samples, metric="precomputed"
            ).fit_predict(prepared["distances"])
            if size
            else np.array([], dtype=int)
        )
    memberships, metrics, groups = [], [], []
    usable_metadata = metadata.filter(pl.Series(observed))
    coordinate_names = np.asarray(coordinate_labels)[prepared["active_columns"]]
    for algorithm, raw_labels in [
        ("K-means", kmeans_labels),
        ("DBSCAN", dbscan_labels),
    ]:
        # Stable numbering by best-ranked member makes tables easier to scan.
        label_order = list(
            dict.fromkeys(int(label) for label in raw_labels if label >= 0)
        )
        label_map = {label: index + 1 for index, label in enumerate(label_order)}
        names = [
            "Noise" if label < 0 else f"Cluster {label_map[int(label)]}"
            for label in raw_labels
        ]
        all_labels = np.full(metadata.height, "Unclusterable", dtype=object)
        all_labels[observed] = names
        x, y = np.full(metadata.height, np.nan), np.full(metadata.height, np.nan)
        x[observed], y[observed] = prepared["coordinates"].T
        memberships.append(
            metadata.with_columns(
                pl.lit(algorithm).alias("algorithm"),
                pl.Series("cluster", all_labels, dtype=pl.String),
                pl.Series("svd_1", x),
                pl.Series("svd_2", y),
            )
        )
        retained = raw_labels >= 0
        cluster_count = len(label_order)
        silhouette = None
        if 1 < cluster_count < int(retained.sum()):
            with threadpool_limits(limits=2):
                silhouette = float(
                    silhouette_score(
                        prepared["distances"][np.ix_(retained, retained)],
                        raw_labels[retained],
                        metric="precomputed",
                    )
                )
        metrics.append(
            {
                "algorithm": algorithm,
                "selected": metadata.height,
                "usable": size,
                "unclusterable": int((~observed).sum()),
                "clusters": cluster_count,
                "clustered": int(retained.sum()),
                "noise": int((~retained).sum()),
                "coverage": float(retained.mean()) if size else None,
                "silhouette": silhouette,
                "settings": (
                    f"k={effective_k} (requested {requested_k})"
                    if algorithm == "K-means"
                    else f"ε={eps:.2f}; min_samples={min_samples}"
                ),
            }
        )
        for label in [*label_order, *([-1] if np.any(~retained) else [])]:
            positions = np.flatnonzero(raw_labels == label)
            member_table = usable_metadata[positions.tolist()]
            common = ""
            if label >= 0:
                frequencies = np.asarray(
                    prepared["binary"][positions].mean(axis=0)
                ).ravel()
                top_coordinates = np.argsort(-frequencies, kind="stable")[:5]
                common = "; ".join(
                    f"{coordinate_names[index]} ({frequencies[index]:.0%})"
                    for index in top_coordinates
                    if frequencies[index] > 0
                )
            groups.append(
                {
                    "algorithm": algorithm,
                    "cluster": "Noise"
                    if label < 0
                    else f"Cluster {label_map[label]}",
                    "members": len(positions),
                    "highest_ranked_members": "; ".join(
                        member_table["label"].head(5).to_list()
                    ),
                    "common_profile_memberships": common,
                }
            )
    group_schema = {
        "algorithm": pl.String,
        "cluster": pl.String,
        "members": pl.Int64,
        "highest_ranked_members": pl.String,
        "common_profile_memberships": pl.String,
    }
    return {
        "members": pl.concat(memberships),
        "metrics": pl.DataFrame(
            metrics,
            schema_overrides={"coverage": pl.Float64, "silhouette": pl.Float64},
        ),
        "groups": pl.DataFrame(groups, schema=group_schema),
    }


@app.cell(hide_code=True)
def compute_ranked_profile_clustering(
    feature_pagerank_catalog,
    network_node_ids,
    ranked_cluster_account_eps,
    ranked_cluster_feature_eps,
    ranked_cluster_k,
    ranked_cluster_min_samples,
    ranked_cluster_profiles,
):
    ranked_cluster_results = {
        "Accounts": fit_ranked_profile_clusters(
            ranked_cluster_profiles["Accounts"],
            ranked_cluster_k.value,
            ranked_cluster_account_eps.value,
            ranked_cluster_min_samples.value,
            feature_pagerank_catalog["name"].to_numpy(),
        ),
        "Features": fit_ranked_profile_clusters(
            ranked_cluster_profiles["Features"],
            ranked_cluster_k.value,
            ranked_cluster_feature_eps.value,
            ranked_cluster_min_samples.value,
            network_node_ids.astype(str),
        ),
    }
    ranked_cluster_diagnostics = pl.concat(
        [
            _result["metrics"].with_columns(pl.lit(_population).alias("population"))
            for _population, _result in ranked_cluster_results.items()
        ]
    ).select("population", pl.exclude("population"))
    return ranked_cluster_diagnostics, ranked_cluster_results


@app.cell(hide_code=True)
def display_ranked_cluster_diagnostics(ranked_cluster_diagnostics):
    mo.vstack(
        [
            mo.md("### Coverage and separation"),
            mo.ui.table(
                ranked_cluster_diagnostics,
                selection=None,
                show_data_types=False,
                show_column_summaries=False,
                format_mapping={"coverage": "{:.1%}", "silhouette": "{:.3f}"},
            ),
            mo.md(
                "**Coverage** is the fraction of usable profiles assigned to a cluster. "
                "**Silhouette** uses Euclidean distances in the full normalized profile space; "
                "higher values indicate better separation. It is undefined without at least two "
                "clusters and more assigned profiles than clusters. DBSCAN silhouette excludes "
                "noise, so compare it together with coverage, not as a standalone winner. "
                "K-means k is capped by the number of distinct usable profiles. "
                "The controls are exploratory; no best k or ε has been established."
            ),
        ]
    )
    return


@app.function(hide_code=True)
def display_ranked_cluster_population(
    population, prepared, result, eps, min_samples
):
    """Show both fits in one fixed projection, plus tuning and membership evidence."""
    plot_data = result["members"].filter(pl.col("cluster") != "Unclusterable")
    cluster_count = int(result["metrics"]["clusters"].max())
    cluster_domain = [
        f"Cluster {index}" for index in range(1, cluster_count + 1)
    ] + ["Noise"]
    cluster_colors = [
        f"hsl({(index * 137.508 + 215) % 360:.1f},65%,55%)"
        for index in range(cluster_count)
    ] + ["#94a3b8"]
    charts = []
    for algorithm in ["K-means", "DBSCAN"]:
        chart = (
            alt.Chart(plot_data.filter(pl.col("algorithm") == algorithm))
            .mark_circle(size=70, opacity=0.8, stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X(
                    "svd_1:Q", title="SVD component 1", scale=alt.Scale(zero=False)
                ),
                y=alt.Y(
                    "svd_2:Q", title="SVD component 2", scale=alt.Scale(zero=False)
                ),
                color=alt.Color(
                    "cluster:N",
                    title="Assignment",
                    scale=alt.Scale(domain=cluster_domain, range=cluster_colors),
                    legend=alt.Legend(orient="bottom", columns=4),
                ),
                tooltip=[
                    alt.Tooltip("id:N", title="ID"),
                    alt.Tooltip("label:N", title="Member"),
                    alt.Tooltip("rank:Q", title="PageRank rank", format=",.1f"),
                    alt.Tooltip("pagerank:Q", title="PageRank", format=".5g"),
                    alt.Tooltip(
                        "memberships:Q", title="Recorded memberships", format=",d"
                    ),
                    alt.Tooltip("cluster:N", title="Assignment"),
                ],
            )
            .properties(
                title=f"{population} · {algorithm}", width="container", height=330
            )
            .interactive()
        )
        # Show only labels present in this fit, retaining the shared color mapping.
        present = set(
            plot_data.filter(pl.col("algorithm") == algorithm)["cluster"].to_list()
        )
        chart = chart.encode(
            color=alt.Color(
                "cluster:N",
                title="Assignment",
                scale=alt.Scale(
                    domain=[label for label in cluster_domain if label in present],
                    range=[
                        color
                        for label, color in zip(cluster_domain, cluster_colors)
                        if label in present
                    ],
                ),
                legend=alt.Legend(orient="bottom", columns=4),
            )
        )
        charts.append(
            mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
        )
    size = prepared["profiles"].shape[0]
    neighbor_view = mo.md(
        "Too few usable profiles for the selected minimum-neighbor count."
    )
    if size >= min_samples:
        # The row includes distance zero to self, as DBSCAN min_samples does.
        neighbor_distances = np.sort(
            np.partition(prepared["distances"], min_samples - 1, axis=1)[
                :, min_samples - 1
            ]
        )
        neighbor_data = pl.DataFrame(
            {
                "ordered_profile": np.arange(1, size + 1),
                "neighbor_distance": neighbor_distances,
            }
        )
        curve = (
            alt.Chart(neighbor_data)
            .mark_line(color="#475569")
            .encode(
                x=alt.X(
                    "ordered_profile:Q",
                    title="Usable profiles ordered by neighbor distance",
                ),
                y=alt.Y(
                    "neighbor_distance:Q",
                    title="Distance to required neighbor",
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("ordered_profile:Q"),
                    alt.Tooltip("neighbor_distance:Q", format=".3f"),
                ],
            )
        )
        threshold = (
            alt.Chart(pl.DataFrame({"epsilon": [eps]}))
            .mark_rule(color="#ea580c", strokeDash=[6, 4])
            .encode(
                y="epsilon:Q",
                tooltip=[alt.Tooltip("epsilon:Q", title="ε", format=".2f")],
            )
        )
        neighbor_view = mo.ui.altair_chart(
            (curve + threshold).properties(
                width="container",
                height=190,
                title=f"DBSCAN tuning · ε = {eps:.2f}; minimum neighbors = {min_samples}",
            ),
            chart_selection=False,
            legend_selection=False,
        )
    group_table = result["groups"].rename(
        {
            "common_profile_memberships": "common_features"
            if population == "Accounts"
            else "common_accounts"
        }
    )
    member_tables = {
        algorithm: mo.ui.table(
            result["members"]
            .filter(pl.col("algorithm") == algorithm)
            .drop("algorithm", "svd_1", "svd_2"),
            selection=None,
            pagination=True,
            page_size=10,
            show_data_types=False,
            show_column_summaries=False,
            format_mapping={"pagerank": "{:.6g}", "rank": "{:,.1f}"},
        )
        for algorithm in ["K-means", "DBSCAN"]
    }
    return mo.vstack(
        [
            mo.hstack(charts, widths="equal", align="start", gap=1.5),
            mo.md(
                "Both plots share one 2D SVD projection, fitted only for display; "
                "clustering and silhouette use the full profiles. "
                "Apparent overlap in 2D can hide separation in other dimensions. "
                "Cluster numbers do not establish correspondence between algorithms. "
                "Hover for members, scroll to zoom, and drag to pan."
            ),
            neighbor_view,
            mo.md(
                "The dashed line is ε. Profiles at or below it have enough neighbors "
                "to be DBSCAN core points; other profiles may join as border points or remain noise. "
                "A sharp change near √2 signals weak overlap and sensitivity to radius."
            ),
            mo.md("#### Group summaries"),
            mo.ui.table(
                group_table,
                selection=None,
                pagination=True,
                page_size=10,
                show_data_types=False,
                show_column_summaries=False,
            ),
            mo.md(
                "Groups list their five highest-ranked members. Common memberships show up to "
                "five features (for account clusters) or accounts (for feature clusters), "
                "ordered by within-group prevalence; percentages are the share of group members "
                "with that membership. These are descriptive overlaps, not enrichment scores."
            ),
            mo.accordion(
                {
                    "Browse every assignment (including noise and unclusterable accounts)": mo.ui.tabs(
                        member_tables
                    )
                }
            ),
        ]
    )


@app.cell(hide_code=True)
def display_ranked_profile_clustering(
    ranked_cluster_account_eps,
    ranked_cluster_feature_eps,
    ranked_cluster_min_samples,
    ranked_cluster_profiles,
    ranked_cluster_results,
):
    mo.ui.tabs(
        {
            _population: display_ranked_cluster_population(
                _population,
                ranked_cluster_profiles[_population],
                _result,
                ranked_cluster_account_eps.value
                if _population == "Accounts"
                else ranked_cluster_feature_eps.value,
                ranked_cluster_min_samples.value,
            )
            for _population, _result in ranked_cluster_results.items()
        }
    )
    return


@app.cell(hide_code=True)
def describe_ego_network_selection():
    mo.md(r"""
    ## Select Node to Analyze

    Choose an ego node to construct a reactive egocentric network. Changing the
    dropdown reruns only the dependent cells. The table below contains the selected
    ego plus every account the ego follows. The network plot has its own relationship
    selector: **Following (outgoing)**, **Followers (incoming)**, **Friends (mutual)**,
    or **All** incoming and outgoing neighbors.

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
        full_width=True,
    )
    return (ego_node_dropdown,)


@app.cell(hide_code=True)
def control_ego_network_visualization(
    ego_network_feature,
    ego_network_labels,
    ego_network_neighbor_edges,
    ego_network_node_limit,
    ego_network_relationship,
    ego_node_dropdown,
):
    mo.vstack(
        [
            mo.md("### Network controls"),
            mo.md("#### Nodes and filters"),
            topic_widget_grid(
                [
                    ego_node_dropdown,
                    ego_network_relationship,
                    ego_network_feature,
                    ego_network_node_limit,
                ]
            ),
            mo.md("#### Display options"),
            topic_widget_grid([ego_network_neighbor_edges, ego_network_labels]),
            mo.md(
                "Features are sorted by neighbor count, highest first. Search features recorded "
                "among this ego's incoming or outgoing neighbors. The ego remains visible even "
                "if it does not have the chosen feature. Nodes without recorded features match "
                "only **All features**."
            ),
        ],
        gap=1,
    )
    return


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
            pl.col("node").is_in(_ego_following_series.implode()),
        )
        .with_columns(
            pl.col(RELATIONSHIP_TYPES)
            .list.filter(pl.element().is_in(_ego_following_series.implode()))
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

    mo.ui.table(egocentric_network_lf, selection=None)
    return (ego_node,)


@app.cell(hide_code=True)
def define_ego_network_controls():
    ego_network_relationship = mo.ui.dropdown(
        options=[
            "Following (outgoing)",
            "Followers (incoming)",
            "Friends (mutual)",
            "All",
        ],
        value="Following (outgoing)",
        label="Relationships to include",
        full_width=True,
    )
    ego_network_node_limit = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=80,
        label="Maximum displayed nodes (including ego)",
        full_width=True,
    )
    ego_network_neighbor_edges = mo.ui.checkbox(
        value=True,
        label="Show connections between neighbors",
    )
    ego_network_labels = mo.ui.checkbox(value=False, label="Show all node IDs")
    return (
        ego_network_labels,
        ego_network_neighbor_edges,
        ego_network_node_limit,
        ego_network_relationship,
    )


@app.cell(hide_code=True)
def catalog_ego_network_features(ego_node, feats_lf, nodes_lf):
    _ego_lists = (
        nodes_lf.filter(pl.col("node") == ego_node)
        .select("followers", "following")
        .collect()
        .row(0, named=True)
    )
    _neighbor_ids = pl.Series(
        "node",
        sorted(
            (set(_ego_lists["followers"]) | set(_ego_lists["following"])) - {ego_node}
        ),
        dtype=pl.Int64,
    )
    ego_network_feature_catalog = (
        nodes_lf.filter(pl.col("node").is_in(_neighbor_ids.implode()))
        .select(pl.col("features").list.unique().alias("id"))
        .explode("id")
        .filter(pl.col("id").is_not_null())
        .group_by("id")
        .len(name="neighbors")
        .join(feats_lf.select("id", "name"), on="id", how="left", validate="1:1")
        .sort(["neighbors", "name", "id"], descending=[True, False, False])
        .collect()
    )
    return (ego_network_feature_catalog,)


@app.cell(hide_code=True)
def select_ego_network_feature(ego_network_feature_catalog):
    _feature_options = {"All features": None}
    _feature_options.update(
        {
            f"{row['name']} (ID {row['id']}; {row['neighbors']:,} neighbors)": row["id"]
            for row in ego_network_feature_catalog.iter_rows(named=True)
        }
    )
    ego_network_feature = mo.ui.dropdown(
        options=_feature_options,
        value="All features",
        allow_select_none=False,
        searchable=True,
        full_width=True,
        label="Filter neighbors by node feature",
    )
    return (ego_network_feature,)


@app.cell(hide_code=True)
def collect_ego_network_visualization_data(
    ego_network_feature,
    ego_network_relationship,
    ego_node,
    nodes_lf,
):
    _ego_relationships = (
        nodes_lf.filter(pl.col("node") == ego_node)
        .select("followers", "following")
        .collect()
        .row(0, named=True)
    )
    _ego_followers = set(_ego_relationships["followers"]) - {ego_node}
    _ego_following = set(_ego_relationships["following"]) - {ego_node}
    _mode = ego_network_relationship.value
    _neighbors = {
        "Following (outgoing)": _ego_following,
        "Followers (incoming)": _ego_followers,
        "Friends (mutual)": _ego_followers & _ego_following,
        "All": _ego_followers | _ego_following,
    }[_mode]
    if ego_network_feature.value is not None:
        _candidate_ids = pl.Series("node", sorted(_neighbors), dtype=pl.Int64)
        _neighbors = set(
            nodes_lf.filter(pl.col("node").is_in(_candidate_ids.implode()))
            .filter(pl.col("features").list.contains(ego_network_feature.value))
            .select("node")
            .collect()["node"]
            .to_list()
        )
    _members = pl.Series("members", sorted(_neighbors | {ego_node}), dtype=pl.Int64)
    _followers = pl.Series("followers", sorted(_ego_followers), dtype=pl.Int64)
    _following = pl.Series("following", sorted(_ego_following), dtype=pl.Int64)
    ego_network_visualization_data = (
        nodes_lf.filter(pl.col("node").is_in(_members.implode()))
        .select(
            "node",
            pl.col("following")
            .list.filter(pl.element().is_in(_members.implode()))
            .alias("alter_following"),
            pl.col("followers")
            .list.filter(pl.element().is_in(_members.implode()))
            .list.len()
            .alias("alter_followers_count"),
            pl.col("following")
            .list.filter(pl.element().is_in(_members.implode()))
            .list.len()
            .alias("alter_following_count"),
            pl.col("node").is_in(_followers.implode()).alias("follows_ego"),
            pl.col("node").is_in(_following.implode()).alias("ego_follows"),
        )
        .collect()
    )
    return (ego_network_visualization_data,)


@app.function(hide_code=True)
def build_ego_network_plot_data(
    frame, selected_node, max_nodes, include_neighbor_edges, relationship_mode="All"
):
    """Lay out a bounded induced network; retain original IDs and full ego degrees."""
    ranked = frame.with_columns(
        (pl.col("alter_followers_count") + pl.col("alter_following_count")).alias(
            "ego_network_degree"
        )
    ).sort(["ego_network_degree", "node"], descending=[True, False])
    chosen = [selected_node] + (
        ranked.filter(pl.col("node") != selected_node)
        .head(max_nodes - 1)["node"]
        .to_list()
    )
    chosen_set = set(chosen)
    directed_edges = {
        (row["node"], target)
        for row in ranked.iter_rows(named=True)
        if row["node"] in chosen_set
        for target in row["alter_following"]
        if target in chosen_set
        and target != row["node"]
        and (include_neighbor_edges or selected_node in (row["node"], target))
    }
    # The mode controls ego arrows; neighbor-to-neighbor edges remain an
    # independent option. Classify nodes from the original relationship lists.
    if relationship_mode == "Following (outgoing)":
        directed_edges = {
            (source, target)
            for source, target in directed_edges
            if target != selected_node
        }
    elif relationship_mode == "Followers (incoming)":
        directed_edges = {
            (source, target)
            for source, target in directed_edges
            if source != selected_node
        }
    graph = nx.DiGraph()
    graph.add_nodes_from(chosen)
    graph.add_edges_from(sorted(directed_edges))
    positions = nx.spring_layout(graph.to_undirected(), seed=42, iterations=100)
    metadata = {row["node"]: row for row in ranked.iter_rows(named=True)}
    nodes = []
    for node in chosen:
        role = (
            "Selected ego"
            if node == selected_node
            else "Mutual with ego"
            if metadata[node]["ego_follows"] and metadata[node]["follows_ego"]
            else "Follows ego"
            if metadata[node]["follows_ego"]
            else "Followed by ego"
        )
        nodes.append(
            {
                "node": str(node),
                "x": float(positions[node][0]),
                "y": float(positions[node][1]),
                "role": role,
                "incoming": metadata[node]["alter_followers_count"],
                "outgoing": metadata[node]["alter_following_count"],
                "degree": metadata[node]["ego_network_degree"],
                "size": 350
                if node == selected_node
                else 65 + 10 * np.sqrt(graph.degree(node)),
            }
        )
    links, arrows = [], []
    for source, target in sorted(directed_edges):
        start, end = positions[source], positions[target]
        delta = end - start
        length = float(np.linalg.norm(delta))
        mutual = (target, source) in directed_edges
        color = "#0d9488" if mutual else "#94a3b8"
        edge = {
            "source": str(source),
            "target": str(target),
            "color": color,
            "relationship": "Mutual" if mutual else "One-way",
            "x": float(start[0]),
            "y": float(start[1]),
            "x2": float(end[0]),
            "y2": float(end[1]),
        }
        # Draw each mutual pair once, with a separate arrow for each direction.
        if not mutual or source < target:
            links.append(edge)
        arrow_position = end - delta * min(0.035 / max(length, 1e-9), 0.35)
        arrows.append(
            {
                **edge,
                "x": float(arrow_position[0]),
                "y": float(arrow_position[1]),
                "angle": float(np.degrees(np.arctan2(delta[0], delta[1])) % 360),
            }
        )
    return {
        "nodes": nodes,
        "links": links,
        "arrows": arrows,
        "total_nodes": frame.height,
        "directed_edges": len(directed_edges),
    }


@app.cell(hide_code=True)
def compute_ego_network_layout(
    ego_network_neighbor_edges,
    ego_network_node_limit,
    ego_network_relationship,
    ego_network_visualization_data,
    ego_node,
):
    ego_network_plot_data = build_ego_network_plot_data(
        ego_network_visualization_data,
        ego_node,
        ego_network_node_limit.value,
        ego_network_neighbor_edges.value,
        ego_network_relationship.value,
    )
    return (ego_network_plot_data,)


@app.cell(hide_code=True)
def display_ego_network_visualization(
    ego_network_feature,
    ego_network_labels,
    ego_network_neighbor_edges,
    ego_network_plot_data,
    ego_network_relationship,
    ego_node,
):
    _plot = ego_network_plot_data
    _x = alt.X("x:Q", axis=None, scale=alt.Scale(zero=False, padding=35))
    _y = alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, padding=35))
    _edge_tooltip = [
        alt.Tooltip("source:N", title="From (follower)"),
        alt.Tooltip("target:N", title="To (followed account)"),
        alt.Tooltip("relationship:N", title="Relationship"),
    ]
    _edges = (
        alt.Chart(alt.Data(values=_plot["links"]))
        .mark_rule(
            opacity=0.35,
            strokeWidth=0.8,
        )
        .encode(
            x=_x,
            y=_y,
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color("color:N", scale=None),
            tooltip=_edge_tooltip,
        )
    )
    _arrows = (
        alt.Chart(alt.Data(values=_plot["arrows"]))
        .mark_point(
            shape="triangle-up",
            angle=alt.ExprRef(
                expr="atan2(scale('x', datum.x2) - scale('x', datum.x), scale('y', datum.y) - scale('y', datum.y2)) * 180 / PI"
            ),
            filled=True,
            size=24,
            opacity=0.65,
        )
        .encode(
            x=_x,
            y=_y,
            color=alt.Color("color:N", scale=None),
            tooltip=_edge_tooltip,
        )
    )
    _node_chart = alt.Chart(alt.Data(values=_plot["nodes"]))
    _nodes = _node_chart.mark_circle(stroke="white", strokeWidth=1.3, opacity=1).encode(
        x=_x,
        y=_y,
        size=alt.Size("size:Q", scale=None, legend=None),
        color=alt.Color(
            "role:N",
            title="Node relationship",
            scale=alt.Scale(
                domain=[
                    "Selected ego",
                    "Mutual with ego",
                    "Followed by ego",
                    "Follows ego",
                ],
                range=["#f59e0b", "#0d9488", "#3b82f6", "#a855f7"],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("node:N", title="Node ID"),
            alt.Tooltip("role:N", title="Relationship to ego"),
            alt.Tooltip(
                "incoming:Q", title="Followers in full ego network", format=",d"
            ),
            alt.Tooltip(
                "outgoing:Q", title="Following in full ego network", format=",d"
            ),
        ],
    )
    _labels = (
        _node_chart.transform_filter(
            alt.datum.role == "Selected ego" if not ego_network_labels.value else "true"
        )
        .mark_text(dy=-16, fontSize=10, color="#475569")
        .encode(x=_x, y=_y, text="node:N")
    )
    ego_network_chart = (
        alt.layer(_edges, _arrows, _nodes, _labels)
        .properties(
            width="container",
            height=520,
            title=f"Node {ego_node}: {ego_network_relationship.value}",
        )
        .configure_view(stroke=None)
        .interactive()
    )
    _shown = len(_plot["nodes"])
    _scope = (
        "Connections between neighbors are included."
        if ego_network_neighbor_edges.value
        else "Only connections involving the ego are shown."
    )
    mo.vstack(
        [
            mo.md(
                f"**{ego_network_relationship.value}** · **{_shown:,} of {_plot['total_nodes']:,} nodes** · "
                f"**{_plot['directed_edges']:,} displayed directed edges**. "
                + (
                    "The ego is always retained; remaining nodes are ranked by incoming + outgoing degree within the full ego network. "
                    if _shown < _plot["total_nodes"]
                    else "All nodes in this ego network are displayed. "
                )
                + _scope
            ),
            mo.md(
                f"**Node feature:** {ego_network_feature.selected_key}. "
                + (
                    "No neighbors match this feature and relationship filter; only the ego is shown. "
                    if _plot["total_nodes"] == 1
                    and ego_network_feature.value is not None
                    else ""
                )
                + (
                    "The ego is retained as a reference regardless of its features."
                    if ego_network_feature.value is not None
                    else ""
                )
            ),
            mo.ui.altair_chart(
                ego_network_chart, chart_selection=False, legend_selection=False
            ),
            mo.md(
                "Hover for node IDs and connection counts; scroll to zoom, drag to pan, and double-click to reset. "
                "Arrows point from follower to followed account. Teal edges are mutual; gray edges are one-way. "
                "Larger neighbors have more displayed connections. Positions show network structure, not geography. "
                "The relationship selector chooses neighbors relative to the ego: following shows outgoing ego arrows, "
                "followers shows incoming ego arrows, friends retains both directions for mutual neighbors, "
                "and all includes both sets of neighbors and directions. Connections between neighbors, when enabled, "
                "retain their observed directions. Node colors describe the original relationship to the ego, "
                "even when one arrow direction is hidden. Counts refer to the full selected network before the display limit. "
                "Self-loops are omitted."
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
