# Performance and Operations (PostgreSQL)

## Query Tuning Defaults

- Start from real slow queries, not generic tuning folklore.
- Use `EXPLAIN (ANALYZE, BUFFERS)` for hotspots.
- Index to match filter, join, and order-by patterns.
- Watch bloat, autovacuum posture, and lock contention.

## Operations Defaults

- Backups are only real if restore is tested.
- Monitor connection saturation, replication lag, deadlocks, and slow queries.
- Partition large retention-heavy tables when deletion cost becomes operationally expensive.

## Principal Review Lens

- Which queries dominate p95/p99 latency?
- Is autovacuum helping or falling behind?
- What fails first under traffic spike: CPU, IO, locks, or connections?
