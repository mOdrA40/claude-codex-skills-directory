# Multi-Tenant MongoDB

## Rules

- Isolation strategy should match blast radius and support needs.
- Shard key and tenant distribution are deeply related.
- Noisy-neighbor control requires quota and workload visibility.
- Export, restore, and diagnostics must preserve tenant boundaries.

## Principal Review Lens

- Can one tenant create a hot shard or hot collection?
- What is the blast radius of one filter bug?
- How is tenant-specific recovery handled safely?
