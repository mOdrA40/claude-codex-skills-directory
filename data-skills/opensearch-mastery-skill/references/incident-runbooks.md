# Incident Runbooks (OpenSearch)

## Cover at Minimum

- Heap pressure and GC storms.
- Red or yellow cluster state.
- Ingest backlog.
- Slow search incident.
- Bad mapping or template rollout.
- Snapshot/restore emergency.

## Response Rules

- Stabilize the most business-critical workload first.
- Prefer reversible actions before broad cluster surgery.
- Preserve evidence around shard movement, GC, and query offenders.
- Communicate clearly about freshness, search quality, and data safety separately.

## Principal Review Lens

- Can on-call reduce user pain quickly?
- Which emergency action risks making recovery slower later?
- What evidence proves the cluster is actually healthy again?
- Are runbooks built around real incident patterns?
