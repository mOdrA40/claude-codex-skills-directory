# Memory Management (Redis)

## Rules

- Memory is the primary production budget in Redis.
- Track fragmentation, eviction, allocator behavior, and big keys.
- Maxmemory policy must match business semantics.
- Compression and data model choices matter more than wishful tuning.

## Principal Review Lens

- Are evictions safe or user-visible corruption?
- Which keys dominate memory footprint?
- Is fragmentation hiding the real headroom?
