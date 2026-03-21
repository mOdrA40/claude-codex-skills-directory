# Aggregations and Analytics (Elasticsearch)

## Rules

- Aggregations can become the hidden cost center of a cluster.
- Heavy analytics and user search often conflict on the same nodes.
- Sampling or precomputation may be safer than brute-force aggregations.
- Large cardinality fields need special care.

## Analytics Heuristics

### Aggregations are workload-shaping decisions

Aggregation-heavy use cases change heap pressure, query latency, index design, and whether the cluster is acting more like a search engine or an analytics platform.

### Search and analytics should be costed separately

The platform should know which latency, heap, and query-cost burden comes from user search versus exploratory dashboards, operational analytics, or support investigation queries.

### Precomputation often buys more than brute force

When repeated heavy aggregations dominate cost, derived indices, rollups, or another system may be safer than repeatedly asking the search cluster to do analytic work at query time.

## Common Failure Modes

### Hidden analytics tax

The cluster is described as a search system, but aggregation-heavy workloads are already consuming the real budget.

### Cardinality pain normalized

The team accepts slow or expensive high-cardinality analytics without revisiting whether Elasticsearch is the right runtime path.

### Search users paying analytics cost

Interactive search quality degrades because aggregation workloads compete on the same cluster without clear isolation.

## Principal Review Lens

- Which aggregation drives heap or latency pain?
- Are we mixing analytics and search on the same cluster carelessly?
- Would another system own this workload better?
- Which analytic use case is currently imposing the highest hidden tax on search behavior?
