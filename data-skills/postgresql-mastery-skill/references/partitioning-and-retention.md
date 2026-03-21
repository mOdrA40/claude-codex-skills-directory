# Partitioning and Retention (PostgreSQL)

## Rules

- Partition only when retention, maintenance, or pruning justify it.
- Tie partitioning strategy to dominant query filters.
- Retention should be an operational workflow, not ad-hoc deletes.
- Validate planning and index behavior on partitioned tables.

## Partitioning Heuristics

### Partitioning should buy operational boringness

The strongest justification for partitioning is usually safer retention management, easier archival, better maintenance windows, or predictable pruning on known hot paths.

### Retention is a recurring systems workflow

Deleting old data should be treated as a regular operational motion with clear load expectations, scheduling, and rollback thinking—not as an occasional heroic cleanup.

### Query benefit must be proven, not assumed

Partitioning can create planner and index complexity, so teams should validate whether the intended pruning and maintenance benefits are actually real in production-like behavior.

## Common Failure Modes

### Partitioning by fashion

The schema adopts partitioning because large tables exist, not because maintenance, pruning, or retention workflows truly benefit enough.

### Retention path not operationalized

Old data is supposed to roll off cleanly, but the actual archive/drop process is too manual or too risky under live traffic.

### Planner optimism

The design assumes partition pruning will save the hot queries, yet real execution still scans or coordinates more than expected.

## Principal Review Lens

- Are we partitioning for real operational pain or fashion?
- Does pruning actually help the hottest queries?
- What is the drop/archive procedure during peak load?
- Which partitioning assumption is least proven under real retention and workload pressure?
