# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"], sql_output="native")

with app.setup:
    from pathlib import Path

    import duckdb
    import marimo as mo


@app.cell
def _():
    # Resolve every data location relative to this notebook so it can be run
    # from any working directory.
    ROOT_PATH = Path(__file__).resolve().parent

    DATA_PATH = ROOT_PATH / "data"
    TWITTER_COMBINED_TXT_PATH = DATA_PATH / "twitter_combined.txt"
    PARQUET_OUT_PATH = DATA_PATH / "out"
    return DATA_PATH, PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH


@app.cell
def _(PARQUET_OUT_PATH):
    PARQUET_OUT_PATH.mkdir(parents=True, exist_ok=True)
    raw_db = duckdb.connect()
    duckdb_temp_sql_path = str(PARQUET_OUT_PATH / ".duckdb-tmp").replace("'", "''")
    return duckdb_temp_sql_path, raw_db


@app.cell
def _(raw_db):
    memory_limit_setting = mo.sql(
        f"""
        SET memory_limit = '512MB'
        """,
        output=False,
        engine=raw_db,
    )
    return (memory_limit_setting,)


@app.cell
def _(raw_db):
    thread_setting = mo.sql(
        f"""
        SET threads = 2
        """,
        output=False,
        engine=raw_db,
    )
    return (thread_setting,)


@app.cell
def _(duckdb_temp_sql_path, raw_db):
    temp_directory_setting = mo.sql(
        f"""
        SET temp_directory = '{duckdb_temp_sql_path}'
        """,
        output=False,
        engine=raw_db,
    )
    return (temp_directory_setting,)


@app.cell
def _(memory_limit_setting, raw_db, temp_directory_setting, thread_setting):
    # All SQL cells use the connection after its memory, thread, and spill
    # settings have been applied.
    _ = memory_limit_setting, temp_directory_setting, thread_setting
    db = raw_db
    return (db,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Data

    DuckDB reads the source files with SQL and writes normalized Parquet caches.
    Existing caches are scanned directly, so source text is parsed only when a
    cache is missing.
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


@app.cell
def _(PARQUET_OUT_PATH, TWITTER_COMBINED_TXT_PATH):
    edges_cache_path = PARQUET_OUT_PATH / "edges.parquet"
    edges_cache_exists = edges_cache_path.is_file()
    if not edges_cache_exists and not TWITTER_COMBINED_TXT_PATH.is_file():
        raise FileNotFoundError(TWITTER_COMBINED_TXT_PATH)
    PARQUET_OUT_PATH.mkdir(parents=True, exist_ok=True)
    edges_source_sql_path = str(TWITTER_COMBINED_TXT_PATH).replace("'", "''")
    edges_temp_sql_path = str(edges_cache_path.with_suffix(".tmp.parquet")).replace(
        "'", "''"
    )
    return (
        edges_cache_exists,
        edges_cache_path,
        edges_source_sql_path,
        edges_temp_sql_path,
    )


@app.cell
def _(db, edges_cache_exists, edges_source_sql_path, edges_temp_sql_path):
    edges_cache_copy = mo.sql(
        f"""
        COPY (
            FROM query(
                CASE WHEN {edges_cache_exists}
                    THEN $$SELECT 1 WHERE FALSE$$
                    ELSE $$
                    SELECT DISTINCT column0::BIGINT AS "from", column1::BIGINT AS "to"
                    FROM read_csv(
                        '{edges_source_sql_path}',
                        delim = ' ',
                        header = false,
                        columns = {{'column0': 'BIGINT', 'column1': 'BIGINT'}}
                    )
                    ORDER BY "from", "to"
                    $$
                END
            )
        ) TO '{edges_temp_sql_path}' (FORMAT parquet, COMPRESSION zstd)
        """,
        output=False,
        engine=db,
    )
    return (edges_cache_copy,)


@app.cell
def _(edges_cache_copy, edges_cache_path):
    _ = edges_cache_copy
    _temporary_path = edges_cache_path.with_suffix(".tmp.parquet")
    if edges_cache_path.is_file():
        _temporary_path.unlink(missing_ok=True)
    else:
        _temporary_path.replace(edges_cache_path)
    edges_cache_sql_path = str(edges_cache_path).replace("'", "''")
    return (edges_cache_sql_path,)


@app.cell
def _(db, edges_cache_sql_path):
    edges = mo.sql(
        f"""
        SELECT * FROM read_parquet('{edges_cache_sql_path}')
        """,
        output=False,
        engine=db,
    )
    return


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


@app.cell
def _(DATA_PATH):
    # Discover every ego-network source file recursively. Sorting makes file
    # processing and resulting row order reproducible across operating systems.
    TWITTER_EGO_FILES_PATH = DATA_PATH / "twitter"

    FEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.feat"))
    EGOFEAT_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.egofeat"))
    FEATNAMES_FILES = sorted(TWITTER_EGO_FILES_PATH.rglob("*.featnames"))
    return EGOFEAT_FILES, FEATNAMES_FILES, FEAT_FILES, TWITTER_EGO_FILES_PATH


@app.cell
def _(FEATNAMES_FILES, PARQUET_OUT_PATH, TWITTER_EGO_FILES_PATH):
    feature_names_cache_path = PARQUET_OUT_PATH / "feature_names.parquet"
    feature_names_cache_exists = feature_names_cache_path.is_file()
    if not feature_names_cache_exists:
        if not FEATNAMES_FILES:
            raise FileNotFoundError(
                "No .featnames files found and no feature-name cache exists"
            )
        if any(not _path.stem.isdecimal() for _path in FEATNAMES_FILES):
            raise ValueError("Feature filenames must have numeric ego-node IDs")
    PARQUET_OUT_PATH.mkdir(parents=True, exist_ok=True)
    feature_names_source_sql_path = str(TWITTER_EGO_FILES_PATH / "*.featnames").replace(
        "'", "''"
    )
    feature_names_temp_sql_path = str(
        feature_names_cache_path.with_suffix(".tmp.parquet")
    ).replace("'", "''")
    return (
        feature_names_cache_exists,
        feature_names_cache_path,
        feature_names_source_sql_path,
        feature_names_temp_sql_path,
    )


@app.cell
def _(
    db,
    feature_names_cache_exists,
    feature_names_source_sql_path,
    feature_names_temp_sql_path,
):
    feature_names_cache_copy = mo.sql(
        f"""
        COPY (
            FROM query(
                CASE WHEN {feature_names_cache_exists}
                    THEN $$SELECT 1 WHERE FALSE$$
                    ELSE $$
                    SELECT
                        regexp_extract(filename, '([0-9]+)\\.featnames$', 1)::BIGINT AS ego_node,
                        regexp_extract(raw_line, '^\\s*([0-9]+)\\s+(.+?)\\s*$', 1)::BIGINT AS "index",
                        regexp_extract(raw_line, '^\\s*([0-9]+)\\s+(.+?)\\s*$', 2) AS name
                    FROM read_csv(
                        '{feature_names_source_sql_path}',
                        delim = '\\x1f',
                        header = false,
                        columns = {{'raw_line': 'VARCHAR'}},
                        filename = true
                    )
                    WHERE trim(raw_line) <> ''
                    ORDER BY ego_node, "index"
                    $$
                END
            )
        ) TO '{feature_names_temp_sql_path}' (FORMAT parquet, COMPRESSION zstd)
        """,
        output=False,
        engine=db,
    )
    return (feature_names_cache_copy,)


@app.cell
def _(feature_names_cache_copy, feature_names_cache_path):
    _ = feature_names_cache_copy
    _temporary_path = feature_names_cache_path.with_suffix(".tmp.parquet")
    if feature_names_cache_path.is_file():
        _temporary_path.unlink(missing_ok=True)
    else:
        _temporary_path.replace(feature_names_cache_path)
    feature_names_cache_sql_path = str(feature_names_cache_path).replace("'", "''")
    return (feature_names_cache_sql_path,)


@app.cell
def _(db, feature_names_cache_sql_path):
    feats = mo.sql(
        f"""
        SELECT * FROM read_parquet('{feature_names_cache_sql_path}')
        """,
        output=False,
        engine=db,
    )
    return (feats,)


@app.cell
def _(EGOFEAT_FILES, FEAT_FILES, PARQUET_OUT_PATH, TWITTER_EGO_FILES_PATH):
    node_vectors_cache_path = PARQUET_OUT_PATH / "node_feature_vectors.parquet"
    node_vectors_cache_exists = node_vectors_cache_path.is_file()
    if not node_vectors_cache_exists:
        _feature_paths = sorted([*FEAT_FILES, *EGOFEAT_FILES])
        if not _feature_paths:
            raise FileNotFoundError(
                f"No .feat or .egofeat files in {TWITTER_EGO_FILES_PATH}"
            )
        if any(not _path.stem.isdecimal() for _path in _feature_paths):
            raise ValueError("Feature filenames must have numeric ego-node IDs")
    PARQUET_OUT_PATH.mkdir(parents=True, exist_ok=True)
    features_feat_sql_path = str(TWITTER_EGO_FILES_PATH / "*.feat").replace("'", "''")
    features_egofeat_sql_path = str(TWITTER_EGO_FILES_PATH / "*.egofeat").replace(
        "'", "''"
    )
    node_vectors_temp_sql_path = str(
        node_vectors_cache_path.with_suffix(".tmp.parquet")
    ).replace("'", "''")
    return (
        features_egofeat_sql_path,
        features_feat_sql_path,
        node_vectors_cache_exists,
        node_vectors_cache_path,
        node_vectors_temp_sql_path,
    )


@app.cell(hide_code=True)
def _(
    db,
    features_egofeat_sql_path,
    features_feat_sql_path,
    node_vectors_cache_exists,
    node_vectors_temp_sql_path,
):
    node_vectors_cache_copy = mo.sql(
        f"""
        COPY (
            FROM query(
                CASE WHEN {node_vectors_cache_exists}
                    THEN $$SELECT 1 WHERE FALSE$$
                    ELSE $$
                    WITH source_lines AS (
                        SELECT
                            filename,
                            regexp_split_to_array(trim(raw_line), '\\s+') AS tokens
                        FROM read_csv(
                            ['{features_feat_sql_path}', '{features_egofeat_sql_path}'],
                            delim = '\\x1f',
                            header = false,
                            columns = {{'raw_line': 'VARCHAR'}},
                            filename = true
                        )
                        WHERE trim(raw_line) <> ''
                    )
                    SELECT
                        regexp_extract(filename, '([0-9]+)\\.(?:feat|egofeat)$', 1)::BIGINT AS ego_node,
                        CASE
                            WHEN ends_with(filename, '.egofeat')
                                THEN regexp_extract(filename, '([0-9]+)\\.egofeat$', 1)::BIGINT
                            ELSE tokens[1]::BIGINT
                        END AS node,
                        ends_with(filename, '.egofeat') AS is_ego,
                        list_transform(
                            CASE
                                WHEN ends_with(filename, '.egofeat') THEN tokens
                                ELSE tokens[2:]
                            END,
                            feature -> feature::TINYINT
                        ) AS features
                    FROM source_lines
                    $$
                END
            )
        ) TO '{node_vectors_temp_sql_path}' (FORMAT parquet, COMPRESSION zstd)
        """,
        output=False,
        engine=db,
    )
    return (node_vectors_cache_copy,)


@app.cell
def _(node_vectors_cache_copy, node_vectors_cache_path):
    _ = node_vectors_cache_copy
    _temporary_path = node_vectors_cache_path.with_suffix(".tmp.parquet")
    if node_vectors_cache_path.is_file():
        _temporary_path.unlink(missing_ok=True)
    else:
        _temporary_path.replace(node_vectors_cache_path)
    node_vectors_cache_sql_path = str(node_vectors_cache_path).replace("'", "''")
    return (node_vectors_cache_sql_path,)


@app.cell
def _(db, node_vectors_cache_sql_path):
    node_feature_vectors = mo.sql(
        f"""
        SELECT * FROM read_parquet('{node_vectors_cache_sql_path}')
        """,
        output=False,
        engine=db,
    )
    return (node_feature_vectors,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Save Data to Parquet

    The loading cells write missing caches under **`data/out`** with Zstandard
    compression. Existing files are reused. Downstream SQL queries scan the
    Parquet files, allowing DuckDB to push down filters and columns.

    Cached datasets:

    - `edges.parquet`: deduplicated and sorted directed edges
    - `feature_names.parquet`: feature descriptions keyed by ego and index
    - `node_feature_vectors.parquet`: binary feature lists for alters and egos
    - `nodes.parquet`: aggregated follower, following, and friend lists

    The expanded feature view is derived in SQL when needed.

    > To rebuild after changing source files or preprocessing logic, delete the
    > affected file from `data/out` and rerun the loading section. Rebuild
    > `nodes.parquet` whenever `edges.parquet` changes.
    """)
    return


@app.cell
def _(
    db,
    edges_cache_sql_path,
    feature_names_cache_sql_path,
    node_vectors_cache_sql_path,
    nodes_cache_sql_path,
):
    parquet_cache_report = mo.sql(
        f"""
        SELECT 'edges.parquet' AS file, count(*) AS rows
        FROM read_parquet('{edges_cache_sql_path}')
        UNION ALL
        SELECT 'feature_names.parquet', count(*)
        FROM read_parquet('{feature_names_cache_sql_path}')
        UNION ALL
        SELECT 'node_feature_vectors.parquet', count(*)
        FROM read_parquet('{node_vectors_cache_sql_path}')
        UNION ALL
        SELECT 'nodes.parquet', count(*)
        FROM read_parquet('{nodes_cache_sql_path}')
        """,
        output=False,
        engine=db,
    )
    return (parquet_cache_report,)


@app.cell
def _(parquet_cache_report):
    mo.ui.table(parquet_cache_report)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Consolidated Feature View

    The cached `node_feature_vectors` keeps one compact binary list per node.
    The next SQL cell expands each list into feature values and joins local
    indices to their descriptions.

    The expanded view can be large. Query it with a filter on `ego_node`, `node`,
    or `value`. The displayed table previews only 100 compact vectors.
    """)
    return


@app.cell
def _(db, feats, node_feature_vectors):
    node_features = mo.sql(
        f"""
        SELECT
            vectors.ego_node,
            vectors.node,
            vectors.is_ego,
            expanded.ordinal - 1 AS "index",
            feats.name,
            expanded.value
        FROM node_feature_vectors AS vectors
        CROSS JOIN UNNEST(vectors.features) WITH ORDINALITY AS expanded(value, ordinal)
        LEFT JOIN feats
            ON vectors.ego_node = feats.ego_node
            AND expanded.ordinal - 1 = feats."index"
        """,
        output=False,
        engine=db,
    )
    return


@app.cell
def _(db, node_feature_vectors):
    node_vectors_preview = mo.sql(
        f"""
        SELECT * FROM node_feature_vectors LIMIT 100
        """,
        output=False,
        engine=db,
    )
    return (node_vectors_preview,)


@app.cell(hide_code=True)
def _(node_vectors_preview):
    mo.ui.table(node_vectors_preview, selection=None)
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


@app.cell
def _(PARQUET_OUT_PATH):
    nodes_cache_path = PARQUET_OUT_PATH / "nodes.parquet"
    nodes_cache_exists = nodes_cache_path.is_file()
    nodes_temp_sql_path = str(nodes_cache_path.with_suffix(".tmp.parquet")).replace(
        "'", "''"
    )
    return nodes_cache_exists, nodes_cache_path, nodes_temp_sql_path


@app.cell
def _(db, edges_cache_sql_path, nodes_cache_exists, nodes_temp_sql_path):
    nodes_cache_copy = mo.sql(
        f"""
        COPY (
            FROM query(
                CASE WHEN {nodes_cache_exists}
                    THEN $$SELECT 1 WHERE FALSE$$
                    ELSE $$
                    WITH followers AS (
                        SELECT
                            "to" AS node,
                            count(*) AS followers_count,
                            list("from" ORDER BY "from") AS followers
                        FROM read_parquet('{edges_cache_sql_path}')
                        GROUP BY "to"
                    ),
                    following AS (
                        SELECT
                            "from" AS node,
                            count(*) AS following_count,
                            list("to" ORDER BY "to") AS following
                        FROM read_parquet('{edges_cache_sql_path}')
                        GROUP BY "from"
                    ),
                    combined AS (
                        SELECT
                            coalesce(followers.node, following.node) AS node,
                            coalesce(followers.followers_count, 0) AS followers_count,
                            coalesce(following.following_count, 0) AS following_count,
                            coalesce(followers.followers, []::BIGINT[]) AS followers,
                            coalesce(following.following, []::BIGINT[]) AS following
                        FROM followers
                        FULL OUTER JOIN following USING (node)
                    ),
                    mutual AS (
                        SELECT *, list_sort(list_intersect(followers, following)) AS friends
                        FROM combined
                    )
                    SELECT
                        node,
                        followers_count,
                        following_count,
                        list_count(friends) AS friends_count,
                        followers,
                        following,
                        friends
                    FROM mutual
                    $$
                END
            )
        ) TO '{nodes_temp_sql_path}' (FORMAT parquet, COMPRESSION zstd)
        """,
        output=False,
        engine=db,
    )
    return (nodes_cache_copy,)


@app.cell
def _(nodes_cache_copy, nodes_cache_path):
    _ = nodes_cache_copy
    _temporary_path = nodes_cache_path.with_suffix(".tmp.parquet")
    if nodes_cache_path.is_file():
        _temporary_path.unlink(missing_ok=True)
    else:
        _temporary_path.replace(nodes_cache_path)
    nodes_cache_sql_path = str(nodes_cache_path).replace("'", "''")
    return (nodes_cache_sql_path,)


@app.cell
def _(db, nodes_cache_sql_path):
    nodes = mo.sql(
        f"""
        SELECT * FROM read_parquet('{nodes_cache_sql_path}')
        """,
        output=False,
        engine=db,
    )
    return (nodes,)


@app.cell
def _(db, nodes):
    nodes_preview = mo.sql(
        f"""
        SELECT * FROM nodes LIMIT 100
        """,
        output=False,
        engine=db,
    )
    return (nodes_preview,)


@app.cell(hide_code=True)
def _(nodes_preview):
    mo.ui.table(nodes_preview, selection=None)
    return


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


@app.cell
def _(db, nodes):
    node_ids = mo.sql(
        f"""
        SELECT node FROM nodes ORDER BY node
        """,
        output=False,
        engine=db,
    )
    return (node_ids,)


@app.cell
def _(node_ids):
    ego_nodes = [row[0] for row in node_ids.fetchall()]
    if not ego_nodes:
        raise ValueError("The edge dataset contains no nodes")
    ego_node_dropdown = mo.ui.dropdown(
        options=ego_nodes,
        value=ego_nodes[0],
        label="Choose an ego node",
    )

    ego_node_dropdown
    return (ego_node_dropdown,)


@app.cell
def _(ego_node_dropdown):
    ego_node = int(ego_node_dropdown.value)
    return (ego_node,)


@app.cell
def _(db, ego_node, nodes):
    egocentric_network = mo.sql(
        f"""
        WITH network AS (
            SELECT list_distinct(list_concat([node], following)) AS members
            FROM nodes
            WHERE node = {ego_node}
        ),
        selected AS (
            SELECT
                nodes.*,
                list_filter(followers, member -> list_contains(network.members, member)) AS alter_followers,
                list_filter(following, member -> list_contains(network.members, member)) AS alter_following,
                list_filter(friends, member -> list_contains(network.members, member)) AS alter_friends
            FROM nodes
            CROSS JOIN network
            WHERE list_contains(network.members, nodes.node)
        )
        SELECT
            node,
            followers_count,
            followers,
            list_count(alter_followers) AS alter_followers_count,
            alter_followers,
            following_count,
            following,
            list_count(alter_following) AS alter_following_count,
            alter_following,
            friends_count,
            friends,
            list_count(alter_friends) AS alter_friends_count,
            alter_friends
        FROM selected
        ORDER BY node
        """,
        output=False,
        engine=db,
    )
    return (egocentric_network,)


@app.cell
def _(db, egocentric_network):
    egocentric_network_preview = mo.sql(
        f"""
        SELECT * FROM egocentric_network LIMIT 100
        """,
        output=False,
        engine=db,
    )
    return (egocentric_network_preview,)


@app.cell(hide_code=True)
def _(egocentric_network_preview):
    mo.ui.table(egocentric_network_preview, selection=None)
    return


if __name__ == "__main__":
    app.run()
