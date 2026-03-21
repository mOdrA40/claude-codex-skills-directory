---
name: spark-principal-engineer
description: |
  Principal/Senior-level Spark playbook for batch and streaming architecture, data layout, shuffle control, cost-efficient compute, reliability, and operating large-scale data processing platforms.
  Use when: designing Spark workloads, tuning jobs, optimizing cluster economics, or operating Spark platforms in production.
---

# Spark Mastery (Senior → Principal)

## Operate

- Start from data volume, compute economics, shuffle behavior, and correctness requirements.
- Treat Spark as a distributed execution system with real storage, network, and scheduling tradeoffs.
- Prefer explicit workload design over vague “big data” assumptions.
- Optimize for predictable cost, reliability, and debuggable pipelines.

## Default Standards

- Data layout and partitioning must match workload reality.
- Shuffle-heavy patterns require scrutiny.
- Memory and executor tuning should follow evidence.
- Streaming and batch semantics must be separated clearly.
- Platform cost and job performance should be evaluated together.

## References

- Job architecture and dataflow design: [references/job-architecture-and-dataflow-design.md](references/job-architecture-and-dataflow-design.md)
- Partitioning, shuffle, and data layout: [references/partitioning-shuffle-and-data-layout.md](references/partitioning-shuffle-and-data-layout.md)
- Memory, executors, and runtime tuning: [references/memory-executors-and-runtime-tuning.md](references/memory-executors-and-runtime-tuning.md)
- Structured streaming and stateful semantics: [references/structured-streaming-and-stateful-semantics.md](references/structured-streaming-and-stateful-semantics.md)
- Lakehouse integration, storage, and table formats: [references/lakehouse-integration-storage-and-table-formats.md](references/lakehouse-integration-storage-and-table-formats.md)
- Multi-tenant governance and cost control: [references/multi-tenant-governance-and-cost-control.md](references/multi-tenant-governance-and-cost-control.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
