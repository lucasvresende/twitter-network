# Twitter network analysis ideas

This catalog describes analyses supported by the consolidated Twitter SNAP
dataset, suitable visualizations, and recommended tools. It is a roadmap of
possible work, rather than a report of completed analyses.

The consolidated caches inspected on 2026-10-19 contain:

- **81,306 nodes**
- **1,768,149 directed edges**
- **216,817 distinct feature names**
- **70,923 nodes with at least one observed active feature**

There are also **973 circle files** available for later integration. The
notebook currently consolidates edges, node relationships, and features; circle
memberships still need to be loaded for circle-based analyses.

The recommended starting stack is **Polars + python-igraph + SciPy + Altair**,
with **NetworkX for ego networks** and **Plotly for interactive exploration**.

## Whole-network structure

| Analysis | What we can learn | Visualization | Best tools |
| --- | --- | --- | --- |
| Network summary | Node/edge counts, density, self-loops, isolates, reciprocity | Metric cards and summary table | Polars; igraph |
| Degree distributions | How followers, following, and mutual connections vary | Histograms, empirical cumulative distribution functions (ECDFs), log-log complementary CDFs (CCDFs) | Polars + Altair |
| Connection inequality | Whether a small fraction of nodes receives most connections | Lorenz curves, Gini coefficient, top-percentile shares | NumPy + Altair |
| Weakly/strongly connected components | Whether the network forms one large structure or many disconnected groups | Component-size distribution | igraph |
| Core–periphery structure | Which nodes belong to increasingly cohesive cores | Core-size curves; degree versus coreness | igraph; explicitly defined undirected projection |
| Clustering and triangles | How often neighbors connect to each other | Clustering distributions; clustering versus degree | igraph; NetworkX for subgraphs |
| Directed motifs | Prevalence of mutual pairs, chains, cycles, and other three-node patterns | Triad census bars; enrichment heatmap | igraph; NetworkX on smaller graphs |
| Reachability and distances | How far apart reachable accounts are | Sampled shortest-path histogram; reachable-fraction curve | igraph or SciPy sparse graph routines |
| Degree assortativity | Whether highly connected accounts connect to similarly connected accounts | Binned endpoint-degree heatmaps | Polars + SciPy; igraph |

## Individual nodes and ego networks

| Analysis | What we can learn | Visualization | Best tools |
| --- | --- | --- | --- |
| Node profile | Followers, following, mutuals, features, community, percentile ranks | Profile cards and comparison table | Polars + marimo |
| Prominence rankings | Who ranks highly by degree, PageRank, or HITS authority/hub scores | Ranked bars; ranking comparison plots | igraph + Altair |
| Brokerage | Which nodes connect otherwise separated regions | Betweenness rankings; highlighted local graph | Sampled betweenness where supported; NetworkX for ego networks |
| Accessibility | Which nodes can reach others, or be reached, efficiently | Inward/outward harmonic-centrality comparisons | igraph; NetworkX locally |
| Ego-network structure | Whether a node's neighborhood is dense, fragmented, or highly reciprocal | Local network diagram; alter-only statistics | NetworkX + Plotly |
| Structural holes | Whether a node connects redundant contacts or distinct groups | Constraint/effective-size scatterplot | NetworkX on ego networks |
| Pair comparison | Shared followers, shared following, mutuals, feature overlap, connecting paths | Overlap table and highlighted paths | Polars; NetworkX |
| Similar-node discovery | Accounts with similar neighborhoods or features | Ranked similarity table; small similarity graph | SciPy sparse + Polars |
| Unusual profiles | High following but low incoming attention, unusual reciprocity, inconsistent centrality rankings | Scatterplots with highlighted outliers | Polars + SciPy |

PageRank and HITS measure **structural prominence** in this dataset; they do not
establish real-world influence. igraph provides a broad centrality toolkit,
while NetworkX exposes configurable and sampled centrality methods. See the
[igraph tutorial](https://python.igraph.org/en/main/tutorial.html) and
[NetworkX centrality reference](https://networkx.org/documentation/stable/reference/algorithms/centrality.html).

## Communities and social circles

| Analysis | What we can learn | Visualization | Best tools |
| --- | --- | --- | --- |
| Community detection | Groups with comparatively dense internal connections | Community-size bars; community-colored subgraphs | igraph, using Leiden or another suitable method |
| Community connectivity | Which groups connect heavily to one another | Aggregated community graph; mixing heatmap | Polars + igraph + Altair |
| Community characteristics | Differences in degree, reciprocity, clustering, and features | Faceted distributions; feature-enrichment heatmap | Polars + SciPy + Altair |
| Cross-community brokers | Accounts connecting multiple communities | Participation versus within-community connectivity | Polars + igraph |
| Circle overlap | Whether supplied social circles overlap or nest | UpSet-style plot; overlap matrix | Polars + Matplotlib |
| Detected communities versus circles | Whether structural groups resemble the supplied memberships | Agreement table; membership heatmap | igraph + custom overlap-aware evaluation |
| Community stability | Whether results persist across seeds and resolution choices | Stability curves; co-membership heatmap on a subset | igraph + NumPy |

For community detection, explicitly choose a **directed objective**, an
**undirected projection**, or the **mutual-only network**. Those answer different
questions. Confirm that the selected algorithm and installed version support
the chosen graph and objective. Circle memberships can overlap, so ordinary
partition agreement scores alone may be insufficient. See the
[igraph community methods](https://python.igraph.org/en/latest/api/igraph.community.html).

## Features, correlations, and statistical questions

| Analysis | What we can learn | Visualization | Best tools |
| --- | --- | --- | --- |
| Feature coverage | Which nodes have observed active features, and how coverage varies | Coverage bars; feature-count distribution | Polars + Altair |
| Feature popularity | Common and rare features; concentration of feature usage | Ranked bars; frequency CCDF | Polars |
| Feature co-occurrence | Features that frequently appear together | Filtered association heatmap; feature graph | SciPy sparse + Polars |
| Feature similarity and links | Whether connected nodes share more features than comparable unconnected nodes | Similarity distributions; effect-size plot | SciPy sparse + matched-pair sampling |
| Homophily | Whether edges disproportionately connect similar nodes | Observed-versus-null comparisons | Polars + graph-aware randomization |
| Metric correlations | Relationships among degree, reciprocity, PageRank, clustering, and feature count | Spearman correlation heatmap; scatterplot matrix | SciPy + Altair/Plotly |
| Degree mixing | Whether popular accounts follow popular accounts, or accounts that follow many others | Four directed in/out-degree mixing views | Polars + SciPy |
| Group differences | Whether communities or feature-defined groups differ in connectivity | ECDFs, boxplots, effect sizes with intervals | SciPy; statsmodels for adjusted models |
| Adjusted associations | Whether feature associations remain after accounting for degree and coverage | Coefficient plots; partial-effect curves | statsmodels |
| Distribution fitting | Whether degree tails resemble power-law, lognormal, or other distributions | Empirical versus fitted CCDFs | SciPy; specialized fitting if warranted |

Start correlations with **Spearman**, distributions, and effect sizes. Network
observations share nodes and edges, so significance testing needs suitable
network null models or justified resampling units. Ordinary
independent-observation tests can exaggerate evidence. When testing many
features or associations, account for multiple comparisons.

Feature co-occurrence should use sparse matrices, frequency filters, and
selected pairs: a complete matrix across 216,817 features is impractical.
See the [SciPy sparse documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html).

A straight line on a log-log degree plot is insufficient evidence for a
power-law distribution. Fit candidate tails and compare alternatives before
drawing that conclusion.

## Prediction and experiments

| Analysis | What we can learn | Visualization | Best tools |
| --- | --- | --- | --- |
| Missing-link ranking | Which absent relationships appear structurally plausible | Candidate table; precision–recall curve | Direction-aware neighborhood scores; personalized PageRank |
| Feature-assisted link prediction | Whether features improve predictions beyond topology | Baseline-versus-model performance curves | SciPy sparse + scikit-learn |
| Robustness experiments | How connectivity changes after random or targeted node removal | Largest-component size versus removal fraction | igraph + NumPy |
| Diffusion simulations | How hypothetical spreading behaves under specified assumptions | Cascade-size distributions; simulation curves | NumPy + graph adjacency |
| Null-model comparisons | Whether clustering, reciprocity, or mixing exceeds an appropriate randomized baseline | Observed values against null distributions | Degree-preserving graph randomization |

With this static dataset, link prediction evaluates **held-out observed edges**,
not future follows. Missing edges may reflect incomplete collection. Compute
topological predictors on the training graph to avoid leaking held-out edges
into the features, and document how negative candidates are sampled. Diffusion
would be a simulation, not measured information spread.

## Tool choices

| Tool | Role in this notebook |
| --- | --- |
| **Polars** | Aggregations, joins, feature frequencies, node tables, chart preparation. Keep these lazy where possible. |
| **python-igraph** | First choice for whole-network components, centralities, communities, and structural metrics. |
| **NetworkX** | Flexible ego-network work, readable prototypes, and specialized local metrics. |
| **rustworkx** | Alternative compiled graph engine when its supported algorithms fit a measured bottleneck; benchmark against igraph on the actual workload. |
| **SciPy sparse + NumPy** | Sparse adjacency/features, similarity calculations, numerical summaries, simulations. |
| **Altair** | Default for linked statistical charts and faceted comparisons; preaggregate large inputs in Polars. |
| **Plotly** | Hoverable node comparisons and interactive local network plots; WebGL scatter for larger point sets. |
| **Matplotlib** | Static figures, custom statistical plots, and exportable report graphics. |
| **Graphviz** | Carefully labeled small graphs and community-level diagrams; `sfdp` is an option for larger layouts. |
| **Datashader** | Rasterized overviews when plotting many nodes or edges. |
| **statsmodels / scikit-learn** | Adjusted statistical models and predictive baselines, respectively. |
| **marimo** | Reactive node selectors, comparison controls, linked tables, and analysis explanations. |

Relevant documentation:

- [Polars lazy optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/)
- [rustworkx overview and scope](https://www.rustworkx.org/)
- [Altair large-data guidance](https://altair-viz.github.io/user_guide/large_datasets.html)
- [Plotly performance guidance](https://plotly.com/python/performance/)
- [Graphviz sfdp layout](https://www.graphviz.org/docs/layouts/sfdp/)
- [Datashader network rendering](https://datashader.org/user_guide/Networks.html)

## Analysis conventions and performance

- Counts describe the **observed dataset**, rather than complete Twitter
  follower totals.
- Preserve edge direction; document every undirected projection.
- Separate absent feature observations from observed zeros where the source
  permits. Consolidated features currently represent **active in any
  observation**. An empty feature list does not distinguish missing observations
  from observed absence.
- Offer following-based, follower-based, mutual-only, and union ego
  neighborhoods. The current notebook uses the ego plus accounts it follows.
  Measure alter-only density separately from density including the ego.
- Define reciprocity denominators explicitly. Reciprocal directed edges and
  mutual node pairs count the same relationships differently.
- Define treatment of disconnected pairs when reporting distances and
  accessibility. Report reachability alongside distances among reachable pairs.
- Cache expensive metrics and layouts. Sample costly distance and betweenness
  calculations, and record sample sizes and random seeds.
- Build graph objects from the edge list and explicitly add the full node
  universe so feature-only nodes are preserved. Keep an ID-to-index mapping
  when a library uses contiguous vertex indices.
- Use sparse representations for adjacency and features. Avoid dense
  all-pairs similarity or distance matrices.
- Use community summaries and local graph views for exploration; reserve
  full-network rendering for aggregate overviews.
- Treat unusual profiles as candidates for investigation, rather than evidence
  of bots or other behavioral labels unsupported by this dataset.
- Temporal growth, churn, and actual cascade analyses require timestamps or
  longitudinal observations that the consolidated tables do not contain.

## Suggested implementation order

1. Degree distributions and connection inequality.
2. Components and reciprocity.
3. Interactive node profile and ego-network explorer.
4. PageRank and community detection.
5. Metric correlations and community comparisons.
6. Feature similarity and homophily.
7. Circle loading, overlap, and community comparisons.
8. Link prediction, robustness experiments, and null-model comparisons.

This sequence delivers useful descriptive results early and builds reusable
node metrics, graph representations, and visual components for the more
expensive analyses.
