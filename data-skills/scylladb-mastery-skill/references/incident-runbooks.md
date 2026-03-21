# Incident Runbooks (ScyllaDB)

## Cover at Minimum

- Hot shard or hot partition incident.
- Node loss or streaming pressure.
- Tombstone/compaction latency spike.
- Repair backlog or correctness concern.
- Cross-DC degradation.
- Tenant-driven overload.

## Incident Heuristics

### Separate shard-local pain from cluster-wide pain

Operators should first ask whether the dominant issue is:

- one shard or one partition shape
- node-level imbalance
- repair or maintenance pressure
- storage / compaction behavior
- cross-DC or topology-wide degradation

### Protect latency-sensitive workloads explicitly

ScyllaDB is often chosen for low-latency expectations, so runbooks should make it clear which workloads get protected first when contention or maintenance pain rises.

### Recovery must be workload-aware

A calmer cluster average is not enough if one tenant, one partition class, or one latency-critical workload remains unhealthy.

## Response Rules

- Stabilize critical traffic and correctness first.
- Prefer targeted throttling and isolation over broad cluster panic.
- Preserve evidence around shard pressure, repair, and storage behavior.
- Communicate clearly about latency, consistency, and recovery tradeoffs.

## Common Failure Modes

### Average-metric blindness

Cluster-wide averages improve while a shard-local or tenant-local incident continues hurting real users.

### Maintenance pressure mistaken for random instability

The symptoms look chaotic, but the deeper cause is repair, compaction, or workload interaction that was never modeled clearly in operator playbooks.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks future correctness pain?
- What confirms the cluster is healthy again beyond surface metrics?
- Are runbooks aligned with actual failure patterns?
- Which ScyllaDB incident still depends too much on expert intuition instead of explicit runbook logic?
