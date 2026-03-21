# Incident Runbooks (Cassandra)

## Cover at Minimum

- Hot partition incident.
- Node loss or streaming pressure.
- Tombstone/compaction-related latency spike.
- Repair backlog or correctness concern.
- Cross-DC impairment.
- Tenant-driven overload.

## Incident Heuristics

### Triage by pressure source first

Operators should determine whether the incident is dominated by:

- partition-level hotspot behavior
- compaction or tombstone pressure
- repair / anti-entropy backlog
- node or cross-DC topology stress
- tenant or workload-class overload

### Protect correctness and availability tradeoffs explicitly

Cassandra incidents often force real choices between latency, consistency expectations, and cluster stability. Runbooks should make those tradeoffs visible.

### Recovery must outlast surface calm

The cluster is not truly recovered if one dashboard is calmer but hotspot, repair, or compaction debt still guarantees a second incident later.

## Response Rules

- Stabilize critical traffic and correctness before chasing perfect balance.
- Prefer targeted isolation and throttling over broad cluster panic.
- Preserve evidence around partitions, repair, and compaction behavior.
- Communicate clearly about consistency, availability, and recovery tradeoffs.

## Common Failure Modes

### Throughput recovery without structural recovery

Latency temporarily improves, but the same data model, hotspot, or tombstone behavior is still loaded and waiting to flare again.

### Repair panic

Teams respond to correctness fear with operational changes that worsen cluster pressure because repair behavior, backlog, or topology interaction is not understood clearly enough.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks future correctness pain?
- What confirms the cluster is healthy again beyond surface metrics?
- Are runbooks aligned with real cluster failure patterns?
- Which Cassandra incident class still lacks a low-regret first-response path?
