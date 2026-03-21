# Clustering and Sentinel (Redis)

## Rules

- HA topology should match failure tolerance and client behavior.
- Client libraries must be tested against failover and slot movement.
- Understand the operational cost of cluster versus single-node simplicity.
- Failover does not remove the need for degraded-mode planning.

## Topology Heuristics

### HA does not equal semantic safety

Redis clustering or Sentinel can improve availability posture, but they do not automatically guarantee that applications handle failover, stale state, slot movement, or cold-cache recovery correctly.

### Client behavior is part of the topology design

The real architecture includes how libraries reconnect, retry, rediscover nodes, and surface failures to callers during topology change.

### Simpler topology often wins unless requirements prove otherwise

A more complex Redis deployment is justified only when the business value of higher availability or scale clearly outweighs the added operational and client-side complexity.

## Common Failure Modes

### HA comfort without failover truth

The platform feels resilient because redundancy exists, but application behavior under failover remains poorly tested or understood.

### Slot movement surprise

Cluster mechanics behave correctly, yet clients or operators are still surprised by the latency, errors, or recovery cost of redistribution.

### Sentinel assumption drift

The team assumes Sentinel solves failover neatly while operational details like warm-up, degraded mode, and client retry behavior remain weak.

## Principal Review Lens

- What client-visible errors appear during failover?
- Is topology complexity justified by the workload?
- Which maintenance event creates the biggest user impact?
- Which Redis topology assumption is least proven under real failover or maintenance pressure?
