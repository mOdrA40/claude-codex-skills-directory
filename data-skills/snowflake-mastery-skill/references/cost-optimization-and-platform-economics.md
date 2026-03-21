# Cost Optimization and Platform Economics

## Rules

- Cost is a first-class architecture concern in Snowflake.
- Warehouse sizing, concurrency, data layout, and sharing choices all affect economics.
- Cost optimization should preserve business value and platform trust.
- Teams need visibility into the cost of their behavior.

## Practical Guidance

- Track high-cost workloads, long-running queries, and idle or oversized warehouses.
- Align cost ownership with domain/team boundaries.
- Optimize around recurring waste before edge-case micro-optimizations.
- Make tradeoffs between latency and cost explicit.

## Economic Heuristics

### Credits reflect architecture, not just runtime noise

High Snowflake cost is often a signal of weak workload isolation, unclear ownership, overly convenient platform defaults, or poor consumer discipline.

### Optimize for value, not cheapest-looking dashboards

The right question is whether spend creates business value at the correct latency and trust level, not whether one warehouse looks smaller this week.

### Governance should make waste legible

A good platform lets teams explain:

- who owns the spend
- what workload justifies it
- which part is avoidable waste
- what policy would reduce it safely

## Common Failure Modes

### Cost blame without ownership clarity

Everyone agrees spend is too high, but no one can map credits to a workload, team, or platform policy failure clearly enough to act.

### Cheap-looking defaults, expensive fleet behavior

Each local decision seems reasonable, but the aggregate effect of warehouse sprawl, weak lifecycle control, and convenience-heavy patterns becomes very expensive.

## Principal Review Lens

- Which workload gives the worst value-per-credit today?
- Are we paying for convenience, lack of governance, or true business need?
- What one platform policy most reduces waste safely?
- Can the team explain cost drivers in plain language?
- Which cost looks operationally acceptable today but economically dangerous at broader adoption?
