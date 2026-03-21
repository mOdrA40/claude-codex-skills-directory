# Reliability and Operations (Redis)

## Operational Defaults

- Monitor memory, eviction rate, hit ratio, replication lag, and latency.
- Pick AOF/RDB posture based on recovery objectives.
- Protect against hot keys and big-key pathologies.
- Know degraded-mode behavior when Redis is slow or unavailable.

## Reliability Rules

- Redis queues and locks are not magic; understand failure semantics.
- Use lease/timeout discipline for coordination patterns.
- Do not quietly rely on Redis durability beyond what config guarantees.

## Principal Review Lens

- If Redis disappears, what breaks and what degrades gracefully?
- Are we using Redis as cache, system of record, or both by accident?
- Which operational metric will warn us first?
