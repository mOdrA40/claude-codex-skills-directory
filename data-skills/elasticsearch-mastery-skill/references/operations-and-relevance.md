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

## Principal Review Lens

- Which shard/layout choice is creating operational tax?
- Are we solving search relevance or observability ingestion with the same defaults?
- Can the team safely reindex when mappings must change?
