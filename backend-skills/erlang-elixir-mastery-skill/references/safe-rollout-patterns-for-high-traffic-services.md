# Safe Rollout Patterns for High-Traffic BEAM Services

## Principles

- canary risky changes when possible
- define rollback triggers before deploy
- ensure old and new nodes can coexist during rollout
- separate consumer and request-serving rollout when needed

## Good Patterns

- halt rollout on mailbox growth, error rate, or queue lag regression
- keep readiness traffic-safe
- stage compatibility changes before cleanup
- protect critical workflows before broad rollout

## Anti-Patterns

- rollout plus incompatible payload changes together
- topology changes with no objective halt signal
- assuming BEAM uptime removes deploy risk
