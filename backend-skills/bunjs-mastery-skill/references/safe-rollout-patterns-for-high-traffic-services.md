# Safe Rollout Patterns for High-Traffic Bun Services

## Principles

- canary risky changes
- keep request-serving and worker rollout separable
- define rollback trigger before deploy
- verify compatibility for handlers, workers, and migrations

## Good Patterns

- small canaries tied to p99 and error signals
- feature-flag optional expensive behavior
- rollout migrations separately from cleanup
- pause rollout when queue lag or dependency failures accelerate

## Anti-Patterns

- deploy + migration + config flip together
- all-or-nothing version assumptions
- rollout with no objective halt signal
