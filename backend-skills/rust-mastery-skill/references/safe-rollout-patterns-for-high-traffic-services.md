# Safe Rollout Patterns for High-Traffic Rust Services

## Principles

- canary risky changes
- define rollback triggers before release
- ensure old and new versions can coexist
- separate worker rollout from request-serving rollout

## Good Patterns

- halt rollout on p99, error rate, or queue lag regression
- keep readiness traffic-safe
- roll out compatibility changes before cleanup
- stage async consumer changes independently when needed

## Anti-Patterns

- synchronized all-instance switching
- rollout plus migration plus config flip at once
- assuming compile-time safety equals rollout safety
