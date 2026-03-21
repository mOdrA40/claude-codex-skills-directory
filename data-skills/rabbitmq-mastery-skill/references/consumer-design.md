# Consumer Design

## Rules

- Consumers must be idempotent and explicit about ack strategy.
- Prefetch should match workload shape and downstream limits.
- Long-running work should not monopolize channels blindly.
- Poison message handling requires a deliberate path.

## Principal Review Lens

- What duplicate delivery behavior is tolerated?
- Does prefetch maximize throughput or just amplify imbalance?
- How are fatal and recoverable failures separated?
