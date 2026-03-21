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

## Reliability Heuristics

### Reliability starts with clear Redis role definition

The system is easier to operate when teams can say clearly whether Redis is acting as cache, coordination primitive, queue, ephemeral store, or some risky combination of all four.

### Operational safety depends on degraded-mode thinking

The strongest Redis systems know what happens when latency rises, data is evicted, failover occurs, or the node restarts with little or no warm state.

### Semantics matter more than feature names

Redis primitives can look convenient, but queues, locks, and derived state all require explicit thinking about replay, expiry, duplication, and partial failure.

## Common Failure Modes

### Role ambiguity by success

Redis starts as a cache, then quietly becomes a semi-critical coordination or state system without governance catching up.

### Reliability claim beyond config reality

Teams talk as if Redis is safe enough for important data or workflows while the chosen persistence and failover posture do not truly support that promise.

### Degraded mode discovered live

The system only learns what breaks, stalls, or lies to users when Redis is already slow or unavailable.

## Principal Review Lens

- If Redis disappears, what breaks and what degrades gracefully?
- Are we using Redis as cache, system of record, or both by accident?
- Which operational metric will warn us first?
- Which Redis role or promise is currently least explicitly owned?
