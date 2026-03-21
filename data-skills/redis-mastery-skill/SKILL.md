---
name: redis-principal-engineer
description: |
  Principal/Senior-level Redis playbook for caching, data structures, persistence tradeoffs, reliability, memory management, and production operations.
  Use when: designing cache strategy, reviewing eviction/persistence settings, debugging latency or memory pressure, or operating Redis in production.
---

# Redis Mastery (Senior → Principal)

## Operate

- Confirm whether Redis is a cache, primary store for ephemeral state, coordination primitive, queue, or all of the above.
- Treat memory, eviction, persistence, and hot key behavior as core architecture decisions.
- Prefer simple data lifecycles with explicit TTL and ownership.
- Design for cache invalidation and degraded-mode behavior up front.

## Default Standards

- Every cache needs a miss strategy and staleness policy.
- Hot keys, large values, and unbounded cardinality are production risks.
- Pick persistence mode based on recovery objectives, not defaults.
- Observe hit ratio, memory fragmentation, latency, and eviction rate.

## References

- Caching and data models: [references/caching-and-data-models.md](references/caching-and-data-models.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Cache invalidation: [references/cache-invalidation.md](references/cache-invalidation.md)
- Key design: [references/key-design.md](references/key-design.md)
- Persistence and recovery: [references/persistence-and-recovery.md](references/persistence-and-recovery.md)
- Memory management: [references/memory-management.md](references/memory-management.md)
- Hot keys and big keys: [references/hot-keys-and-big-keys.md](references/hot-keys-and-big-keys.md)
- Data structures: [references/data-structures.md](references/data-structures.md)
- Distributed locking: [references/distributed-locking.md](references/distributed-locking.md)
- Queues and streams: [references/queues-and-streams.md](references/queues-and-streams.md)
- Clustering and Sentinel: [references/clustering-and-sentinel.md](references/clustering-and-sentinel.md)
- Observability: [references/observability.md](references/observability.md)
- Security and multi-tenant: [references/security-and-multi-tenant.md](references/security-and-multi-tenant.md)
- Capacity planning: [references/capacity-planning.md](references/capacity-planning.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
