# Partitioning and Retention (PostgreSQL)

## Rules

- Partition only when retention, maintenance, or pruning justify it.
- Tie partitioning strategy to dominant query filters.
- Retention should be an operational workflow, not ad-hoc deletes.
- Validate planning and index behavior on partitioned tables.

## Principal Review Lens

- Are we partitioning for real operational pain or fashion?
- Does pruning actually help the hottest queries?
- What is the drop/archive procedure during peak load?
