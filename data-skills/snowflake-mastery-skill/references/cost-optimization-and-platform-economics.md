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

## Principal Review Lens

- Which workload gives the worst value-per-credit today?
- Are we paying for convenience, lack of governance, or true business need?
- What one platform policy most reduces waste safely?
- Can the team explain cost drivers in plain language?
