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

## Performance Heuristics

### Performance work should start from workload reality

The highest-leverage PostgreSQL tuning usually comes from identifying the few query families, tables, or workflows that dominate user pain rather than applying generic database folklore.

### Operations and performance are coupled

Slow queries, bad plans, autovacuum debt, checkpoints, replication lag, and connection pressure often interact. Treating them as isolated concerns hides the real root cause.

### Stability matters more than one-time speed wins

The goal is not one fast explain plan or one successful benchmark. It is a database posture that stays predictable through deploys, growth, maintenance, and traffic spikes.

## Common Failure Modes

### Point optimization without workload thinking

One query improves, but the broader workflow still does too much work or creates too much locking and IO pressure.

### Operational debt hidden behind acceptable medians

Average latency looks tolerable while vacuum lag, checkpoint pain, or replication stress slowly accumulate into the next incident.

### Database blamed for application behavior

The team treats PostgreSQL as the bottleneck while the hotter problem is often transaction design, query choreography, or unnecessary concurrency.

## Principal Review Lens

- Which queries dominate p95/p99 latency?
- Is autovacuum helping or falling behind?
- What fails first under traffic spike: CPU, IO, locks, or connections?
- Which performance assumption is least likely to hold during the next growth step or operational change?
