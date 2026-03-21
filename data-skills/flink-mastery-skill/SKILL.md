---
name: flink-principal-engineer
description: |
  Principal/Senior-level Flink playbook for stream processing architecture, stateful pipelines, event-time correctness, checkpointing, scaling, and operating low-latency data platforms.
  Use when: designing streaming systems, reviewing stateful Flink jobs, tuning event-time pipelines, or operating Flink in production.
---

# Flink Mastery (Senior → Principal)

## Operate

- Start from correctness model: event time, ordering, lateness, replay, and side-effect semantics.
- Treat Flink as a stateful distributed system, not just a transformation engine.
- Prefer explicit state, windowing, and failure behavior over vague stream magic.
- Optimize for correctness first, then latency, then cost.

## Default Standards

- State and checkpoint strategy must be explicit.
- Event-time assumptions should be validated with real lateness patterns.
- Backpressure and operator chaining require observability.
- External side effects must be designed for retries and replay.
- Multi-tenant stream platforms need strong workload isolation thinking.

## References

- Stream architecture and operator design: [references/stream-architecture-and-operator-design.md](references/stream-architecture-and-operator-design.md)
- State, checkpoints, and savepoint safety: [references/state-checkpoints-and-savepoint-safety.md](references/state-checkpoints-and-savepoint-safety.md)
- Event time, watermarks, and lateness correctness: [references/event-time-watermarks-and-lateness-correctness.md](references/event-time-watermarks-and-lateness-correctness.md)
- Backpressure, scaling, and performance tuning: [references/backpressure-scaling-and-performance-tuning.md](references/backpressure-scaling-and-performance-tuning.md)
- Delivery semantics and external side effects: [references/delivery-semantics-and-external-side-effects.md](references/delivery-semantics-and-external-side-effects.md)
- Multi-tenant governance and stream platform safety: [references/multi-tenant-governance-and-stream-platform-safety.md](references/multi-tenant-governance-and-stream-platform-safety.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
