# Incident Runbooks (ClickHouse)

## Cover at Minimum

- Merge backlog and too many parts.
- Disk pressure or storage imbalance.
- Replica degradation.
- Runaway expensive queries.
- Bad backfill or replay.
- Distributed query instability.

## Response Rules

- Stabilize ingestion and critical queries before long-term cleanup.
- Prefer targeted throttling and isolation over broad panic changes.
- Preserve evidence about part growth, merge backlog, and query offenders.
- Communicate clearly when data freshness is degraded versus data correctness at risk.

## Principal Review Lens

- Can on-call reduce blast radius within minutes?
- Which emergency action risks worse merge or replica recovery later?
- What evidence proves the cluster is healthy again?
- Are runbooks aligned with real failure patterns or only vendor docs?
