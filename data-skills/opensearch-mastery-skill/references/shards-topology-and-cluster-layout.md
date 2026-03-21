# Shards, Topology, and Cluster Layout

## Rules

- Shards are not free parallelism; they are an operational budget.
- Layout should reflect data size, recovery goals, and workload isolation needs.
- Oversharding creates long-term pain in memory, recovery, and operator complexity.
- Topology should separate incompatible workloads where justified.

## Capacity Thinking

- Model node loss, rebalance time, and hot shard behavior.
- Plan replica strategy according to read scaling and resilience goals.
- Keep zone, tier, and role assignments understandable to humans.
- Revisit shard count as data and usage evolve.

## Principal Review Lens

- Which index is paying the highest shard tax today?
- How long does recovery take after node or zone loss?
- Are we mixing workloads that deserve separation?
- What topology assumption is least tested right now?
