# Backup, Restore, and Snapshots (Elasticsearch)

## Rules

- Snapshots must be tested, not just configured.
- Restore time and partial-restore options should be known in advance.
- Reindex and snapshot strategies both affect disaster recovery posture.
- Protect snapshot repositories like production data.

## Principal Review Lens

- How long does cluster or index restore really take?
- Can we restore one tenant or one index surgically?
- What recovery path is most brittle today?
