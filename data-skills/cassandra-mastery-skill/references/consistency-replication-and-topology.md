# Consistency, Replication, and Topology

## Rules

- Consistency level is a business and correctness choice, not only a tuning knob.
- Replication and DC topology should reflect failure domains and latency needs.
- Cross-DC deployments need explicit operational playbooks.
- Theorems are not enough; test what your chosen consistency means in practice.

## Practical Guidance

- Align read/write consistency with invariant importance.
- Distinguish local availability goals from global correctness expectations.
- Keep topology and snitch decisions understandable to operators.
- Rehearse failover and degraded-mode behavior with real clients.

## Topology Heuristics

### Consistency level is a user-truth choice

Read and write consistency settings determine what different users or systems may believe during failure, delay, and cross-DC asymmetry—not just throughput and latency.

### Multi-DC complexity must earn its keep

Additional data centers provide resilience and locality possibilities, but they also create more failure modes, more operator burden, and more misunderstood semantics if the team is not explicit.

### Topology should remain explainable under stress

If application and platform teams cannot clearly describe expected behavior during partial DC impairment, the system is already too opaque for safe operation.

## Common Failure Modes

### Consistency cargo cult

Teams choose levels by habit or copied convention without mapping them back to actual business invariants and failure expectations.

### Multi-DC theater

The architecture claims resilience value that the team has neither tested nor operationally internalized.

### Local success, global confusion

The system appears healthy from one location while cross-DC semantics and user truth diverge in dangerous ways.

## Principal Review Lens

- What correctness promise does this consistency level actually provide?
- Which topology assumption is least tested today?
- Can the team explain multi-DC behavior under partial failure?
- Are we choosing settings for semantics or cargo cult?
- Which failure-path truth would most surprise application owners today?
