---
name: elasticsearch-principal-engineer
description: |
  Principal/Senior-level Elasticsearch playbook for index design, search relevance, cluster operations, performance tuning, observability, and production reliability.
  Use when: designing search workloads, reviewing mappings and analyzers, fixing shard/query bottlenecks, or operating Elasticsearch in production.
---

# Elasticsearch Mastery (Senior → Principal)

## Operate

- Start with query patterns, relevance requirements, data freshness, and retention.
- Treat mapping, analyzer, shard count, and indexing strategy as architectural choices.
- Optimize for predictable operations, not demo-search quality only.
- Distinguish logging/observability use-cases from user-facing search needs.

## Default Standards

- Mappings should be explicit on important fields.
- Shards are not free; oversharding is an operational tax.
- Relevance tuning requires representative queries and evaluation, not intuition.
- Observe indexing throughput, search latency, heap pressure, GC, and shard health.

## References

- Indexing and query design: [references/indexing-and-query-design.md](references/indexing-and-query-design.md)
- Operations and relevance: [references/operations-and-relevance.md](references/operations-and-relevance.md)
- Mappings and analyzers: [references/mappings-and-analyzers.md](references/mappings-and-analyzers.md)
- Query DSL and scoring: [references/query-dsl-and-scoring.md](references/query-dsl-and-scoring.md)
- Shards and cluster layout: [references/shards-and-cluster-layout.md](references/shards-and-cluster-layout.md)
- Indexing pipelines: [references/indexing-pipelines.md](references/indexing-pipelines.md)
- Search relevance: [references/search-relevance.md](references/search-relevance.md)
- Aggregations and analytics: [references/aggregations-and-analytics.md](references/aggregations-and-analytics.md)
- ILM and retention: [references/ilm-and-retention.md](references/ilm-and-retention.md)
- Capacity planning: [references/capacity-planning.md](references/capacity-planning.md)
- Security and multi-tenant: [references/security-and-multi-tenant.md](references/security-and-multi-tenant.md)
- Backup, restore, and snapshots: [references/backup-restore-and-snapshots.md](references/backup-restore-and-snapshots.md)
- Observability: [references/observability.md](references/observability.md)
- Logging vs search clusters: [references/logging-vs-search-clusters.md](references/logging-vs-search-clusters.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
