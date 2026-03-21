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

## Tradeoff Heuristics

### Search and analytics create different promises

Product search optimizes relevance, freshness, and user-facing latency. Analytics workloads optimize aggregation breadth, exploratory depth, and operator efficiency. One cluster can serve both, but only with explicit tradeoff ownership.

### Isolation decisions should follow business pain

The right time to split workloads is not merely when metrics look ugly, but when one use case repeatedly harms the trust, latency, or cost posture of the other.

### Ranking logic must remain measurable

Business boosting, personalization, and search-quality tuning should remain visible enough that teams can reason about quality and cost together.

## Common Failure Modes

### One-cluster wishful thinking

Teams keep incompatible workloads together because separation feels operationally expensive, even though the combined tax is already worse.

### Relevance work hidden inside analytics pressure

Search quality problems are blamed on infra generally when the deeper issue is that search and aggregation compete for the same cluster budget.

### Freshness-vs-latency ambiguity

The platform cannot explain which promise to relax first when indexing pressure and search responsiveness conflict.

## Principal Review Lens

- Are we solving two different problems with one cluster default?
- Which workload wins during contention today?
- What relevance decision lacks measurement?
- Would separation of clusters reduce tax materially?
- Which tradeoff is currently being made implicitly instead of governed explicitly?
