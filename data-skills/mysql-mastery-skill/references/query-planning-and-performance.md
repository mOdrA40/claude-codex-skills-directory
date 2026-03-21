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

## Principal Review Lens

- Which query class dominates total pain, not just max latency?
- Are we optimizing isolated SQL or end-to-end workload behavior?
- What estimate error is leading the optimizer astray?
- Which regression would hurt the business fastest if it returned tomorrow?
