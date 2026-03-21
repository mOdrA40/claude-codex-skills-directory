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

## Principal Review Lens

- What correctness promise does this consistency level actually provide?
- Which topology assumption is least tested today?
- Can the team explain multi-DC behavior under partial failure?
- Are we choosing settings for semantics or cargo cult?
