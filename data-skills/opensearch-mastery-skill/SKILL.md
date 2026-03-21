---
name: opensearch-principal-engineer
description: |
  Principal/Senior-level OpenSearch playbook for index architecture, relevance, observability workloads, cluster tuning, security, and operating search/logging platforms at scale.
  Use when: designing search or logging clusters, reviewing mappings and analyzers, tuning cluster performance, or operating OpenSearch in production.
---

# OpenSearch Mastery (Senior → Principal)

## Operate

- Start from workload type: product search, log analytics, security analytics, or mixed.
- Treat mappings, analyzers, shards, retention, and cluster topology as architectural decisions.
- Separate relevance needs from ingest-heavy observability defaults.
- Optimize for predictable operations and recoverability, not demo queries.

## Default Standards

- Important fields should have explicit mappings.
- Shard count is an operational budget.
- Relevance changes must be evaluated, not guessed.
- Retention and ILM/ISM must match business value.
- Search clusters and logging clusters should not be treated as identical by default.

## References

- Mappings and analyzers: [references/mappings-and-analyzers.md](references/mappings-and-analyzers.md)
- Query DSL, filters, and scoring: [references/query-dsl-filters-and-scoring.md](references/query-dsl-filters-and-scoring.md)
- Shards, topology, and cluster layout: [references/shards-topology-and-cluster-layout.md](references/shards-topology-and-cluster-layout.md)
- Ingestion, pipelines, and retention: [references/ingestion-pipelines-and-retention.md](references/ingestion-pipelines-and-retention.md)
- Search relevance and analytics tradeoffs: [references/search-relevance-and-analytics-tradeoffs.md](references/search-relevance-and-analytics-tradeoffs.md)
- Security, tenancy, and governance: [references/security-tenancy-and-governance.md](references/security-tenancy-and-governance.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
