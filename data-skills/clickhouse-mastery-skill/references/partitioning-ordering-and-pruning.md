# Partitioning, Ordering, and Pruning

## Rules

- Partitioning is primarily about retention, maintenance, and scan reduction.
- Ordering keys should serve dominant filters and aggregation paths.
- Over-partitioning creates operational tax and small-part pain.
- Query pruning should be validated with representative workloads, not assumed.

## Common Failure Modes

- Partitioning by overly granular timestamps or tenant dimensions without justification.
- Ordering keys that look intuitive but do not match query reality.
- Too many small parts causing merge overhead and unstable latency.
- Retention and partition drop processes not aligned with actual lifecycle needs.

## Design Heuristics

### Partition for lifecycle first, performance second

Partitioning in ClickHouse is strongest when it supports retention, maintenance, and bounded scan cost without exploding part management complexity.

### Ordering must match the real access path

Ordering keys should help the most repeated and most valuable filter patterns, not just the query examples that looked nice during initial design.

### Pruning assumptions must be tested under real usage

Many designs assume pruning will save them, but real filters, cardinality, and time-range behavior often reveal much worse scan patterns than expected.

## Additional Failure Modes

### Retention-driven design ignored

The schema looks analytically elegant, but partition maintenance and long-term retention operations are awkward or expensive.

### Ordering by intuition

Teams pick a sort key that sounds reasonable, but it does not actually align with the highest-cost workload classes.

## Principal Review Lens

- Which partition choice is paying for itself in production today?
- What query pattern is still scanning too much data?
- Are we using partitioning to compensate for poor workload design?
- How does this design behave after one year of growth?
- What part of this layout will age worst under volume growth and retention pressure?
