# Shards and Cluster Layout (Elasticsearch)

## Rules

- Shards are an operational budget, not free parallelism.
- Layout should reflect data volume, query patterns, and recovery behavior.
- Oversharding is a long-term tax.
- Reallocation behavior matters during incidents and upgrades.

## Layout Heuristics

### Shard count is a recovery decision

Shard strategy affects not just performance, but rebalance time, node loss recovery, operational flexibility, and how much cluster state complexity the team is willing to carry.

### Layout should reflect workload classes

Search-heavy, ingest-heavy, log analytics, and mixed workloads often need different shard and node-layout assumptions.

### Oversharding pain compounds over time

An oversharded cluster may appear manageable early, then become expensive and slow to recover as data volume, node count, and index count all grow.

## Common Failure Modes

### Parallelism superstition

Teams add more shards assuming it always improves performance, while the real effect is more overhead, weaker recovery posture, and harder operations.

### Layout by habit

Cluster topology follows old defaults or copied vendor examples rather than current ingest, query, and retention reality.

### Recovery blind spot

The system looks fine in steady state, but node loss or rolling upgrades reveal that shard movement and recovery cost were never modeled seriously.

## Principal Review Lens

- Which index is paying the highest shard tax?
- How long does recovery take after node loss?
- Is layout driven by evidence or habit?
- Which shard-layout choice looks fine today but becomes dangerous at double current scale?
