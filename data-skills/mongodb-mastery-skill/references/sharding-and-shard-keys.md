# Sharding and Shard Keys (MongoDB)

## Rules

- Shard keys determine scale behavior and pain distribution.
- Avoid keys that create hot shards or poor distribution.
- Query routing should align with shard key usage where possible.
- Re-sharding is possible but operationally expensive.

## Principal Review Lens

- Which tenant, key, or time bucket becomes hottest?
- Are cross-shard operations unavoidable or accidental?
- What happens if distribution skews badly in six months?
