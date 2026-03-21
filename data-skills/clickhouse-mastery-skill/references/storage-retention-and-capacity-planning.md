# Storage, Retention, and Capacity Planning

## Rules

- Capacity planning must include compression, merges, replicas, retention, and backfill behavior.
- Storage tiering should reflect access patterns and incident needs.
- Long retention increases operational complexity even when raw storage seems cheap.
- Query concurrency and ingest growth can break the system before disk is full.

## Practical Guidance

- Forecast growth by event volume, column mix, and retention class.
- Track part count, merge backlog, and disk hot spots, not only overall utilization.
- Model node loss headroom and recovery time.
- Tie retention policy to business value and regulatory requirements.

## Principal Review Lens

- What resource fails first under 2x growth: CPU, disk, or merge capacity?
- Is long retention truly valuable in this local cluster?
- Which table is becoming an outsized capacity risk?
- Are we over-optimizing storage while under-planning recovery?
