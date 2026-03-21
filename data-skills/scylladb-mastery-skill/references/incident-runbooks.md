# Incident Runbooks (ScyllaDB)

## Cover at Minimum

- Hot shard or hot partition incident.
- Node loss or streaming pressure.
- Tombstone/compaction latency spike.
- Repair backlog or correctness concern.
- Cross-DC degradation.
- Tenant-driven overload.

## Response Rules

- Stabilize critical traffic and correctness first.
- Prefer targeted throttling and isolation over broad cluster panic.
- Preserve evidence around shard pressure, repair, and storage behavior.
- Communicate clearly about latency, consistency, and recovery tradeoffs.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks future correctness pain?
- What confirms the cluster is healthy again beyond surface metrics?
- Are runbooks aligned with actual failure patterns?
