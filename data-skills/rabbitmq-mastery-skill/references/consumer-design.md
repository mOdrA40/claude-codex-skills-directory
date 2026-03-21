# Consumer Design

## Rules

- Consumers must be idempotent and explicit about ack strategy.
- Prefetch should match workload shape and downstream limits.
- Long-running work should not monopolize channels blindly.
- Poison message handling requires a deliberate path.

## Consumer Heuristics

### Ack timing expresses business risk

The right moment to ack depends on what side effects are durable, what duplicates are acceptable, and how much redelivery risk the workflow can tolerate.

### Prefetch is not just a throughput knob

Prefetch changes fairness, backlog shape, consumer memory use, downstream pressure, and how quickly one bad message or slow worker can distort the system.

### Consumer classes should differ by workload

Short idempotent tasks, slow external calls, and high-value irreversible effects should not all share the same consumer assumptions.

## Common Failure Modes

### Ack semantics by habit

Teams copy one ack strategy everywhere without proving it matches the actual business side effects.

### Prefetch optimism

Higher prefetch improves local throughput in tests but worsens imbalance, visibility, and recovery behavior in production.

### Poison-message confusion

The system cannot quickly tell whether a failure deserves retry, quarantine, or human intervention.

## Principal Review Lens

- What duplicate delivery behavior is tolerated?
- Does prefetch maximize throughput or just amplify imbalance?
- How are fatal and recoverable failures separated?
- Which consumer design choice would hurt recovery most during a long-lived downstream outage?
