# Shards and Cluster Layout (Elasticsearch)

## Rules

- Shards are an operational budget, not free parallelism.
- Layout should reflect data volume, query patterns, and recovery behavior.
- Oversharding is a long-term tax.
- Reallocation behavior matters during incidents and upgrades.

## Principal Review Lens

- Which index is paying the highest shard tax?
- How long does recovery take after node loss?
- Is layout driven by evidence or habit?
