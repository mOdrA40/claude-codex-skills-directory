# Aggregations and Analytics (Elasticsearch)

## Rules

- Aggregations can become the hidden cost center of a cluster.
- Heavy analytics and user search often conflict on the same nodes.
- Sampling or precomputation may be safer than brute-force aggregations.
- Large cardinality fields need special care.

## Principal Review Lens

- Which aggregation drives heap or latency pain?
- Are we mixing analytics and search on the same cluster carelessly?
- Would another system own this workload better?
