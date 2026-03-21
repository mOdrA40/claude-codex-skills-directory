# Query Planning and Performance

## Rules

- Query tuning starts with workload evidence, not folklore.
- Execution plans, row estimates, join order, and index usage should be reviewed together.
- Hot-path queries deserve explicit ownership and regression detection.
- Query performance must be considered under realistic concurrency.

## Common Failure Modes

- Full scans hidden behind convenient app abstractions.
- Sorting or grouping on paths unsupported by indexes.
- Good plans on small data turning bad on skewed or grown data.
- Connection and lock pressure being mistaken for pure CPU slowness.

## Performance Heuristics

### Tune by workload class, not isolated slow-query screenshots

MySQL performance work should distinguish between:

- hot OLTP reads
- write-heavy transactional paths
- reporting-style reads
- maintenance or background sweeps

Each class degrades differently under concurrency and growth.

### Plan quality and workload behavior interact

A query may look acceptable alone while still hurting the system because it amplifies lock hold time, connection pressure, or repeated round trips in the application flow.

### Stable performance matters more than one lucky plan

The real goal is not one beautiful plan example. It is a query shape that remains healthy as data distribution, selectivity, and concurrency shift.

## Additional Failure Modes

### Query success masking workflow waste

Each SQL statement is individually acceptable, but the overall request path performs too many queries or repeats work unnecessarily.

### Optimizer blame without application accountability

The database is blamed for slowness even when the hotter problem is unstable query shape, ORM usage, or weak access-pattern discipline.

## Principal Review Lens

- Which query class dominates total pain, not just max latency?
- Are we optimizing isolated SQL or end-to-end workload behavior?
- What estimate error is leading the optimizer astray?
- Which regression would hurt the business fastest if it returned tomorrow?
- Which query currently looks fine only because present scale has not stressed its weakest path yet?
