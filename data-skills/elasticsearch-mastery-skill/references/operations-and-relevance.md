# Operations and Relevance (Elasticsearch)

## Operational Defaults

- Monitor heap, GC, indexing rate, merge pressure, and shard imbalance.
- Keep index lifecycle management aligned with retention needs.
- Test cluster behavior under node loss and reallocation.
- Distinguish search cluster tuning from logging cluster tuning.

## Relevance Rules

- Use evaluation sets for ranking quality.
- Boosting without measurement becomes superstition.
- Synonyms, analyzers, and field weights should be versioned deliberately.

## Combined Heuristics

### Relevance quality depends on operational posture too

Search relevance is not only a scoring problem. It also depends on shard layout, indexing freshness, mapping health, and whether the cluster is under enough pressure to distort user experience.

### Operations should preserve search trust

Tuning storage, lifecycle, shard layout, or indexing behavior is only successful if it does not quietly destroy the relevance, freshness, or consistency that users expect.

### Reindex readiness is part of relevance maturity

Teams that take relevance seriously should also be ready for the operational reality that analyzer and mapping improvements often require controlled reindex paths.

## Common Failure Modes

### Ops/relevance split-brain

Platform teams optimize cluster health while search teams optimize ranking, but nobody owns the interactions between the two.

### Relevance improvements blocked by operational fragility

The team knows what ranking or mapping change would help, but the cluster is too fragile to execute the reindex or rollout safely.

### Stable cluster, untrusted results

Infrastructure dashboards look healthy while users still experience stale, inconsistent, or weakly ranked results.

## Principal Review Lens

- Which shard/layout choice is creating operational tax?
- Are we solving search relevance or observability ingestion with the same defaults?
- Can the team safely reindex when mappings must change?
- Which improvement to relevance is currently most constrained by operational weakness?
