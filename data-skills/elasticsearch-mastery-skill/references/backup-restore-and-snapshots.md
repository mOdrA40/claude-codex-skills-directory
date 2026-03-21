# Backup, Restore, and Snapshots (Elasticsearch)

## Rules

- Snapshots must be tested, not just configured.
- Restore time and partial-restore options should be known in advance.
- Reindex and snapshot strategies both affect disaster recovery posture.
- Protect snapshot repositories like production data.

## Recovery Heuristics

### Snapshot posture should reflect recovery reality

A configured snapshot policy is only useful if teams know what can be restored, how long it takes, how partial restores work, and what business surfaces remain broken during that time.

### Surgical restore is often more valuable than full-cluster theory

In many incidents the important question is whether one index, tenant, or time slice can be restored safely without overreacting across the whole platform.

### Repository trust is part of DR trust

A restore strategy is only as good as the security, durability, and operational validity of the snapshot repository behind it.

## Common Failure Modes

### Snapshot checkbox confidence

The platform feels safe because snapshots exist, but nobody has exercised the real restore path under meaningful time pressure.

### Full-restore thinking only

The team can imagine a total restore but lacks clear procedures for the more common surgical recovery scenarios.

### Repository risk ignored

Snapshot storage is treated as background plumbing even though compromise, misconfiguration, or access failure there can invalidate the whole recovery promise.

## Principal Review Lens

- How long does cluster or index restore really take?
- Can we restore one tenant or one index surgically?
- What recovery path is most brittle today?
- Which recovery assumption is currently most weakly tested under realistic incident conditions?
