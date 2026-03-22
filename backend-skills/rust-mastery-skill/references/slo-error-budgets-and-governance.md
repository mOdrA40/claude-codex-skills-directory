# SLO, Error Budgets, and Governance in Rust Services

## Purpose

Type safety does not remove the need for reliability governance. Principal-level Rust services need explicit service objectives, budget discipline, and release behavior that changes when the system becomes unstable.

## Core Principle

If the team cannot say which user outcomes matter most, it cannot make principled tradeoffs about latency, retries, degradation, or rollout risk.

## Good SLI Candidates

- successful completion rate for critical endpoints or workflows
- p95/p99 latency by traffic class
- async backlog age for workflows with eventual completion guarantees
- degraded-mode rate for optional features
- dependency success rate for required downstreams

## Error Budget Questions

- what burn rate should halt rollout?
- what repeated incident class consumes the most budget?
- when should optional features be disabled to protect core paths?
- which queues or background jobs are part of the customer promise?
- what reliability work becomes mandatory when the system is unstable?

## Bad vs Good

```text
❌ BAD
The service tracks many metrics, but releases continue unchanged during rising latency and repeated incident patterns.

✅ GOOD
Critical SLIs are explicit, budget burn is visible, and rollout posture changes when the system drifts out of tolerance.
```

## Principal Heuristics

- Governance should change engineering behavior, not only dashboards.
- Prefer a small number of critical objectives tied to real user pain.
- If no one can say what must be protected first during instability, the SLO model is too weak.
