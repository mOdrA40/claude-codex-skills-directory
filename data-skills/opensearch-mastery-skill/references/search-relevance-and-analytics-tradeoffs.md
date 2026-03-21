# Search Relevance and Analytics Tradeoffs

## Rules

- Product search relevance and log analytics are different optimization problems.
- Aggregations can be as operationally costly as search queries.
- Relevance evaluation should use representative query sets and judgment criteria.
- Mixing workloads requires explicit tradeoff acceptance.

## Practical Guidance

- Benchmark the real top queries and top aggregations separately.
- Keep business ranking logic visible and versioned.
- Consider workload isolation when analytics or search starts hurting the other.
- Treat query latency goals and index freshness goals as competing budgets when necessary.

## Principal Review Lens

- Are we solving two different problems with one cluster default?
- Which workload wins during contention today?
- What relevance decision lacks measurement?
- Would separation of clusters reduce tax materially?
