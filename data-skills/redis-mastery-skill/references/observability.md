# Observability (Redis)

## Rules

- Monitor latency, hit ratio, evictions, memory, fragmentation, and replication state.
- Distinguish healthy misses from cache collapse.
- Big-key and hot-key detection should be routine.
- Correlate Redis pain with downstream database load.

## Observability Heuristics

### Redis signals should be tied to user and fallback risk

The most useful observability tells operators not just that Redis is unhealthy, but whether the application is about to suffer miss storms, stale-state confusion, or downstream overload.

### Separate cache efficiency from platform safety

Hit ratio alone is never enough. Teams should also know whether memory, eviction, replication, hot keys, or fragmentation are making the current state operationally unsafe.

### One key family can matter more than cluster averages

A single hot feature, tenant, or key pattern may create the real incident while overall latency medians still look acceptable.

## Common Failure Modes

### Good dashboard, weak diagnosis

Teams can see Redis is hurting, but not whether the root issue is memory posture, key design, eviction semantics, or application fallback behavior.

### Hit-ratio obsession

The platform celebrates good hit rates while missing the more dangerous signals around correctness, eviction, and downstream collapse risk.

### Cluster-wide calm hiding hot-key pain

Average numbers stay reasonable while one traffic spike or tenant concentrates enough pressure to hurt real users badly.

## Principal Review Lens

- Which metric predicts cascading failure earliest?
- Can we identify hot tenants or hot features quickly?
- Are dashboards telling us correctness risk or just speed?
- Which missing signal would most improve incident detection before fallback systems collapse?
