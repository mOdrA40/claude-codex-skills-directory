# SQL Performance (CockroachDB)

## Rules

- Start from statements that dominate latency or retries.
- Query shape, scan scope, and index choice still matter in distributed SQL.
- Reduce unnecessary round trips and cross-region reads.
- Review plans with locality and contention in mind.

## Performance Heuristics

### Separate SQL-shape cost from topology cost

One query may be slow because it scans too much, uses the wrong index, or joins poorly. Another may be logically fine but still slow because locality and cross-region coordination are fighting the workload.

### Tune by statement class, not anecdotes

Group statements by:

- hot OLTP reads
- write-heavy transactional paths
- cross-region business flows
- background analytical or operational queries

The right optimization differs for each class.

### Watch retries as part of query pain

In CockroachDB, performance degradation may appear not only as raw latency but also as restart pressure, increased contention, and tail instability.

## Common Failure Modes

### Good SQL, poor locality

The statement looks efficient on paper, but leaseholder placement or region access pattern makes the real latency unacceptable.

### Fast median, painful tail

Average performance appears acceptable while retry-heavy or cross-region cases cause user-visible latency spikes.

### Index fix without workload rethink

Teams add indexes until one query improves, but they never revisit whether the broader transaction or access pattern should be redesigned.

## Principal Review Lens

- Is this query slow because of SQL shape or topology placement?
- Are we reading far more rows than needed?
- Which index reduces both latency and contention?
- What statement class would still hurt badly if the cluster doubled in traffic tomorrow?
