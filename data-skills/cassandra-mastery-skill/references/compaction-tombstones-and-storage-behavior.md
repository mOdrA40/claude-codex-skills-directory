# Compaction, Tombstones, and Storage Behavior

## Rules

- Compaction strategy must match workload and data lifecycle.
- Tombstones are not abstract internals; they directly affect reads and operations.
- TTL, deletes, and time-series patterns need careful storage design.
- Storage behavior should be planned before it becomes an emergency.

## Practical Guidance

- Pick compaction strategy based on update pattern, retention, and read profile.
- Monitor tombstone growth, disk use, and compaction backlog.
- Avoid patterns that create long-lived tombstone pain across large partitions.
- Align retention and delete behavior with compaction expectations.

## Principal Review Lens

- Which table is most likely to suffer tombstone-related incidents?
- Are we using TTL in a way that creates long-term operational tax?
- What compaction assumption is least validated by real workload?
- Which storage problem will surface first as data grows?
