---
name: airflow-principal-engineer
description: |
  Principal/Senior-level Airflow playbook for DAG architecture, scheduling reliability, dependency management, platform governance, and operating workflow orchestration safely at scale.
  Use when: designing DAG ecosystems, reviewing orchestration patterns, scaling schedulers and executors, or operating Airflow in production.
---

# Airflow Mastery (Senior → Principal)

## Operate

- Start from orchestration semantics, data dependencies, and failure ownership.
- Treat Airflow as workflow control infrastructure, not as a place to hide business logic.
- Prefer DAGs that remain readable, recoverable, and operationally predictable.
- Optimize for reliability, clear retries, and low scheduler toil.

## Default Standards

- DAG boundaries should reflect real workflow ownership.
- Tasks should be idempotent and retry-aware.
- Scheduling semantics must match data availability and SLA reality.
- Platform governance should control DAG sprawl and dependency chaos.
- Airflow should orchestrate systems, not become the system.

## References

- DAG architecture and task boundary design: [references/dag-architecture-and-task-boundary-design.md](references/dag-architecture-and-task-boundary-design.md)
- Scheduling, SLAs, and dependency semantics: [references/scheduling-slas-and-dependency-semantics.md](references/scheduling-slas-and-dependency-semantics.md)
- Executors, workers, and platform scaling: [references/executors-workers-and-platform-scaling.md](references/executors-workers-and-platform-scaling.md)
- Retries, idempotency, and external side effects: [references/retries-idempotency-and-external-side-effects.md](references/retries-idempotency-and-external-side-effects.md)
- Governance, multi-team DAG ecosystems, and ownership: [references/governance-multi-team-dag-ecosystems-and-ownership.md](references/governance-multi-team-dag-ecosystems-and-ownership.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
