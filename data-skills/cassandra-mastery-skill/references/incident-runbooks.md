# Incident Runbooks (Cassandra)

## Cover at Minimum

- Hot partition incident.
- Node loss or streaming pressure.
- Tombstone/compaction-related latency spike.
- Repair backlog or correctness concern.
- Cross-DC impairment.
- Tenant-driven overload.

## Response Rules

- Stabilize critical traffic and correctness before chasing perfect balance.
- Prefer targeted isolation and throttling over broad cluster panic.
- Preserve evidence around partitions, repair, and compaction behavior.
- Communicate clearly about consistency, availability, and recovery tradeoffs.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks future correctness pain?
- What confirms the cluster is healthy again beyond surface metrics?
- Are runbooks aligned with real cluster failure patterns?
