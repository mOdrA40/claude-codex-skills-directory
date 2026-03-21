# Incident Runbooks (OpenSearch)

## Cover at Minimum

- Heap pressure and GC storms.
- Red or yellow cluster state.
- Ingest backlog.
- Slow search incident.
- Bad mapping or template rollout.
- Snapshot/restore emergency.

## Incident Heuristics

### Separate ingestion pain from query pain

Some incidents primarily damage freshness and indexing, while others damage search latency, cluster stability, or relevance trust. Runbooks should classify that quickly.

### Protect cluster recovery options first

The safest first moves are usually those that reduce pressure without making shard recovery, heap stability, or mapping rollback harder later.

### Recovery must include search trust

A cluster is not truly healthy if it is green but relevance, freshness, or index correctness remains materially degraded for users.

## Response Rules

- Stabilize the most business-critical workload first.
- Prefer reversible actions before broad cluster surgery.
- Preserve evidence around shard movement, GC, and query offenders.
- Communicate clearly about freshness, search quality, and data safety separately.

## Common Failure Modes

### Green-state complacency

Cluster health indicators look better, but ingest delay, query regressions, or mapping damage still undermine user trust.

### Generic search response to mapping incident

Teams treat the issue like ordinary slow search when the real blast radius came from schema or template decisions.

## Principal Review Lens

- Can on-call reduce user pain quickly?
- Which emergency action risks making recovery slower later?
- What evidence proves the cluster is actually healthy again?
- Are runbooks built around real incident patterns?
- Which OpenSearch incident still depends too much on expert intuition rather than explicit playbook steps?
