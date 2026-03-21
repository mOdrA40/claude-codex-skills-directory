---
name: loki-principal-engineer
description: |
  Principal/Senior-level Loki playbook for log architecture, label design, retention, query performance, tenancy, and operating cost-efficient log platforms at scale.
  Use when: designing logging platforms, reviewing label strategy, tuning queries, managing retention, or operating Loki in production.
---

# Loki Mastery (Senior → Principal)

## Operate

- Start from operator and security questions, not from collecting every log forever.
- Treat labels, retention, and query posture as architecture decisions.
- Prefer cost-aware log design with explicit tenant and compliance boundaries.
- Optimize for debuggability without creating cardinality or storage disaster.

## Default Standards

- Labels must stay bounded and operationally meaningful.
- Query behavior should be predictable under multi-team load.
- Retention must match value, compliance, and incident needs.
- Multi-tenant controls should be explicit.
- Logging should support incidents, audits, and debugging without becoming chaos.

## References

- Log architecture and label strategy: [references/log-architecture-and-label-strategy.md](references/log-architecture-and-label-strategy.md)
- Query patterns and performance: [references/query-patterns-and-performance.md](references/query-patterns-and-performance.md)
- Retention, storage, and cost control: [references/retention-storage-and-cost-control.md](references/retention-storage-and-cost-control.md)
- Multi-tenant governance and security: [references/multi-tenant-governance-and-security.md](references/multi-tenant-governance-and-security.md)
- Ingestion pipelines and agents: [references/ingestion-pipelines-and-agents.md](references/ingestion-pipelines-and-agents.md)
- Correlation with metrics and traces: [references/correlation-with-metrics-and-traces.md](references/correlation-with-metrics-and-traces.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
