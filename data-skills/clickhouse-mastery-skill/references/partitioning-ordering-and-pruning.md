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

## Principal Review Lens

- Which partition choice is paying for itself in production today?
- What query pattern is still scanning too much data?
- Are we using partitioning to compensate for poor workload design?
- How does this design behave after one year of growth?
