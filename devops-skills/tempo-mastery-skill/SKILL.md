---
name: tempo-principal-engineer
description: |
  Principal/Senior-level Tempo playbook for trace architecture, storage strategy, sampling-aware operations, tenant governance, and operating large-scale trace backends.
  Use when: designing trace storage, correlating observability workflows, tuning Tempo, or operating trace platforms in production.
---

# Tempo Mastery (Senior → Principal)

## Operate

- Start from what traces must answer in production and how long that truth needs to remain available.
- Treat trace ingestion, storage, search patterns, and tenancy as platform design decisions.
- Prefer simple trace backend operations over feature sprawl.
- Optimize for trustworthy trace availability during incidents.

## Default Standards

- Trace retention should align with debugging and compliance value.
- Multi-tenant access and cost boundaries must be explicit.
- Correlation with metrics/logs should guide UX and storage choices.
- Backend design should account for sampling and burst ingest patterns.
- The platform must make missing traces diagnosable.

## References

- Trace storage architecture and block lifecycle: [references/trace-storage-architecture-and-block-lifecycle.md](references/trace-storage-architecture-and-block-lifecycle.md)
- Ingestion, batching, and scale behavior: [references/ingestion-batching-and-scale-behavior.md](references/ingestion-batching-and-scale-behavior.md)
- Query patterns and search workflows: [references/query-patterns-and-search-workflows.md](references/query-patterns-and-search-workflows.md)
- Correlation with metrics, logs, and exemplars: [references/correlation-with-metrics-logs-and-exemplars.md](references/correlation-with-metrics-logs-and-exemplars.md)
- Multi-tenant governance and cost control: [references/multi-tenant-governance-and-cost-control.md](references/multi-tenant-governance-and-cost-control.md)
- Retention, storage, and durability: [references/retention-storage-and-durability.md](references/retention-storage-and-durability.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
