# Sharding and Shard Keys (MongoDB)

## Rules

- Shard keys determine scale behavior and pain distribution.
- Avoid keys that create hot shards or poor distribution.
- Query routing should align with shard key usage where possible.
- Re-sharding is possible but operationally expensive.

## Shard-Key Heuristics

### Choose keys by workload shape, not abstract distribution alone

A shard key that distributes data nicely on paper may still be poor if dominant queries, writes, or tenant access patterns do not align with it.

### Think about future skew early

Shard-key decisions should anticipate:

- tenant growth concentration
- time-based surges
- write-heavy operational paths
- query patterns that will emerge later

### Re-sharding cost should shape caution today

Because changing shard strategy later is expensive, the initial decision should be reviewed like a platform architecture choice, not a schema detail.

## Common Failure Modes

### Distribution without operability

The key spreads data acceptably but still creates costly cross-shard queries, awkward routing, or hard-to-explain hot paths.

### Time-bucket optimism

A key choice looks good initially, then time-correlated traffic makes one shard range disproportionately painful.

### Tenant skew denial

One tenant or cohort grows into the dominant write or query source, and the original shard strategy never anticipated it.

## Principal Review Lens

- Which tenant, key, or time bucket becomes hottest?
- Are cross-shard operations unavoidable or accidental?
- What happens if distribution skews badly in six months?
- What shard-key assumption will age worst if the business succeeds?
