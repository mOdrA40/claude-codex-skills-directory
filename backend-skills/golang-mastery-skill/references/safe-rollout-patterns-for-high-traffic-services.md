# Safe Rollout Patterns for High-Traffic Go Services

## Principles

- prefer canaries and staged rollouts
- define rollback triggers before release
- ensure old and new binaries can coexist
- separate worker rollout from request-serving rollout

## Good Patterns

- halt rollout on p99, error rate, or queue lag regression
- keep readiness traffic-safe
- roll out migrations in expand/contract sequence
- protect critical traffic before broad rollout

## Anti-Patterns

- synchronized all-instance switching
- rollout plus data migration plus feature flag at once
- scaling out instead of halting bad rollout
