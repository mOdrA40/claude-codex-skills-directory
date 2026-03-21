# Memory Management (Redis)

## Rules

- Memory is the primary production budget in Redis.
- Track fragmentation, eviction, allocator behavior, and big keys.
- Maxmemory policy must match business semantics.
- Compression and data model choices matter more than wishful tuning.

## Memory Heuristics

### Memory policy is product policy

If Redis can evict or drop valuable state, then memory configuration is directly shaping user experience and business correctness.

### Measure real memory ownership

Teams should know whether the main pressure comes from:

- one tenant or cohort
- one feature or key family
- fragmentation and allocator behavior
- large values or unbounded cardinality

### Bigger nodes do not fix weak memory models

Without better TTL discipline, key design, and workload governance, more memory often only delays the same incident.

## Common Failure Modes

### Eviction policy mismatch

The configured policy is operationally convenient but semantically unsafe for the kind of state Redis is currently holding.

### Headroom illusion

Teams think they have safe margin while fragmentation, big keys, or uneven tenant pressure are already consuming the real headroom.

### Shared-keyspace governance failure

Memory growth is blamed on Redis generally, but the true issue is that no one governs which feature or tenant may consume how much of the cache budget.

## Principal Review Lens

- Are evictions safe or user-visible corruption?
- Which keys dominate memory footprint?
- Is fragmentation hiding the real headroom?
- What memory consumer should be redesigned before the next growth step makes recovery painful?
