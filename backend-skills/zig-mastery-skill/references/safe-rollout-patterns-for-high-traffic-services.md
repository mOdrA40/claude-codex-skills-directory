# Safe Rollout Patterns for High-Traffic Zig Services

## Principles

- canary risky releases
- define rollback triggers before deploy
- ensure old and new binaries can coexist when compatibility matters
- separate compatibility rollout from cleanup

## Good Patterns

- halt rollout on memory pressure, queue age, error rate, or dependency regression
- keep readiness traffic-safe
- stage worker changes separately when they affect backlog semantics
- protect critical routes before broad rollout

## Anti-Patterns

- deploy plus irreversible cleanup together
- rollout with no visibility into allocator or queue pressure
- assume low-level control automatically means safe release behavior
