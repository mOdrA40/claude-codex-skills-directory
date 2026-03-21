# Capacity Planning (Redis)

## Rules

- Capacity is mostly about memory, hot traffic, and persistence cost.
- Plan headroom for failover, fragmentation, and warm-up.
- Growth models should include key cardinality explosion.
- Benchmark with realistic value sizes and access distributions.

## Capacity Heuristics

### Memory headroom should include recovery behavior

A Redis deployment is only truly well-sized if it can survive failover, warm-up, eviction churn, and peak traffic without turning one memory problem into a wider application incident.

### Worst-case key shape matters more than average value size

Capacity plans should account for hot tenants, big values, unbounded cardinality, and key families that can grow asymmetrically.

### Persistence and replication change the budget

The cost of AOF, RDB, replicas, and failover is part of real capacity posture, not an afterthought on top of memory estimates.

## Common Failure Modes

### Average memory optimism

The platform seems comfortably sized until one hot key family, eviction burst, or failover reveals that real headroom was much thinner than expected.

### Warm-up cost ignored

The cluster can restart, but repopulation load and cold-cache behavior create more downstream pain than teams modeled.

### Cardinality explosion surprise

Key count or per-tenant growth accelerates faster than the sizing model assumed, and capacity pain appears suddenly.

## Principal Review Lens

- What breaks first under 2x traffic: memory, latency, or eviction?
- How much safe headroom exists during failover?
- Are growth forecasts based on real key distributions?
- Which capacity assumption is least likely to survive a failover plus traffic spike event?
