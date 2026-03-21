# Read/Write Performance and Query Patterns

## Rules

- Query performance must be evaluated with real partition distributions and concurrency.
- Cassandra rewards predictable access patterns and punishes ad-hoc querying.
- Write-heavy success can hide later read-side pain if partitioning is weak.
- Tune around workload shape, not benchmark mythology.

## Performance Guidance

- Track p95/p99 latency per operation type and partition class.
- Understand coordinator cost, read repair implications, and hot partition behavior.
- Avoid features or query patterns that fight the storage model.
- Measure under node loss or compaction pressure too.

## Principal Review Lens

- Which query pattern is paying the highest hidden cost?
- Are we optimizing happy-path latency while ignoring tail behavior?
- What workload looks fine until compaction or repair begins?
- Which design choice most constrains future scale?
