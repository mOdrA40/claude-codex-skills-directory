# Vacuum and Bloat (PostgreSQL)

## Rules

- Autovacuum posture is part of application design on write-heavy systems.
- Bloat, dead tuples, and freeze risk should be measured early.
- Hot tables deserve explicit monitoring and tuning.
- UPDATE-heavy patterns can become silent IO taxes.

## Operational Heuristics

### Treat bloat as workload feedback

Bloat is often a sign that table design, update pattern, retention strategy, or vacuum posture no longer matches the workload reality.

### Hot tables deserve dedicated attention

The few tables absorbing the most updates or churn can dominate storage waste, IO cost, and maintenance pain for the whole system.

### Recovery requires more than vacuum folklore

Some situations need deeper changes in batching, retention, partitioning, fillfactor, or schema behavior—not just “run more vacuum.”

## Common Failure Modes

### Autovacuum faith without evidence

Teams assume defaults are fine without checking whether the actual write pattern, table churn, and freeze pressure justify that confidence.

### Bloat normalized as background tax

Storage and IO inefficiency gradually become accepted until latency, maintenance windows, or failover behavior become painful.

### Write pattern denial

The application keeps generating churn-heavy updates while the database is blamed for the resulting bloat and vacuum pressure.

## Principal Review Lens

- Which tables are accumulating dead tuples fastest?
- Is autovacuum keeping up with write pressure?
- Are we using the database in a way that guarantees bloat pain?
- What workload or table design should change before tuning vacuum any further?
