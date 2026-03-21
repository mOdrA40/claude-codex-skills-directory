# Compaction, Tombstones, and Storage Management

## Rules

- Storage behavior should be designed around TTL, deletes, and data lifecycle explicitly.
- Tombstones and compaction are central to operational stability.
- Pick compaction strategy based on workload and retention, not habit.
- Disk and I/O planning must include maintenance cost.

## Practical Guidance

- Monitor tombstone density, compaction load, and disk hot spots.
- Avoid patterns that create chronic tombstone pain.
- Align retention and delete behavior with storage engine expectations.
- Keep table-specific storage risk visible to operators.

## Principal Review Lens

- Which table is most likely to create a storage incident?
- Are TTL choices creating hidden long-term tax?
- What compaction assumption is least proven by production data?
- Which storage change would improve cluster safety most?
