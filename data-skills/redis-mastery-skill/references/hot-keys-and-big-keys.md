# Hot Keys and Big Keys (Redis)

## Rules

- Hot keys create latency cliffs and uneven scaling.
- Big keys create memory, CPU, and operational pain.
- Detect them before incidents via telemetry and sampling.
- Mitigation usually needs design changes, not only bigger boxes.

## Failure Model

Hot keys and big keys often stay invisible until one peak event, one tenant, or one product launch turns them into tail-latency and cost incidents.

Common causes include:

- one shared cache key serving too much traffic
- one tenant dominating a shared key shape
- collection-like values growing without review
- expensive serialization or mutation against large values

## Practical Heuristics

### Design key ownership explicitly

If a single key represents too much shared state, the system is likely building a hotspot by design.

### Treat big keys as operational debt

Even if they work today, big keys often complicate eviction, replication, failover, and debugging.

### Measure worst-case tenant behavior

Average workload shape does not protect you from one tenant or event dominating a shared cache pattern.

## Common Failure Modes

### Peak-only pain

The system seems fine in normal traffic, then one campaign or one spike makes a hot key dominate CPU, bandwidth, or latency.

### Hidden blast radius in shared keyspace

One product surface or tenant behavior degrades many others because the hottest keys are too globally shared.

### Bigger-box optimism

Teams throw memory or larger nodes at the problem while the key design keeps recreating the same pain.

## Principal Review Lens

- Which key becomes hottest during peak business events?
- What key shape creates worst-case serialization cost?
- Can one tenant dominate a shared keyspace?
- What key should be split, sharded, or redesigned before the next traffic spike?
