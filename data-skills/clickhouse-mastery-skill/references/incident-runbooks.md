# Incident Runbooks (ClickHouse)

## Cover at Minimum

- Merge backlog and too many parts.
- Disk pressure or storage imbalance.
- Replica degradation.
- Runaway expensive queries.
- Bad backfill or replay.
- Distributed query instability.

## Incident Heuristics

### Classify by dominant storage or query pressure

Operators should quickly determine whether the main issue is:

- part explosion / merge backlog
- query-cost spike
- replica degradation
- storage pressure
- replay or backfill overload

### Protect critical freshness and critical queries separately

Some incidents hurt ingestion freshness first, others hurt interactive analytics first. Runbooks should distinguish which promise is currently failing.

### Recovery must include compaction reality

The cluster is not healthy again if foreground behavior improves temporarily while background merges, part counts, or storage strain still guarantee another flare-up.

## Response Rules

- Stabilize ingestion and critical queries before long-term cleanup.
- Prefer targeted throttling and isolation over broad panic changes.
- Preserve evidence about part growth, merge backlog, and query offenders.
- Communicate clearly when data freshness is degraded versus data correctness at risk.

## Common Failure Modes

### Query calm, storage debt alive

User-facing queries improve for a moment, but merge backlog and part pressure remain structurally unhealthy.

### Replica incident treated as generic slowness

The team responds to symptoms without understanding whether replication or distributed-query behavior is the true limiter.

## Principal Review Lens

- Can on-call reduce blast radius within minutes?
- Which emergency action risks worse merge or replica recovery later?
- What evidence proves the cluster is healthy again?
- Are runbooks aligned with real failure patterns or only vendor docs?
- Which ClickHouse failure mode still lacks a low-regret first response?
