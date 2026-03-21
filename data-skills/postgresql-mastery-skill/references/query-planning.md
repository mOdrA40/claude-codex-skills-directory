# Query Planning (PostgreSQL)

## Defaults

- Use `EXPLAIN (ANALYZE, BUFFERS)` on real hotspots.
- Optimize for stable query shape, not one lucky plan.
- Review row estimates, join order, scan type, and sort cost together.
- Favor query clarity over ORM-generated surprises on hot paths.

## Planning Heuristics

### Bad plans often start with bad assumptions

When PostgreSQL chooses a poor plan, the issue is often stale statistics, misleading predicates, skewed data distribution, or query patterns that hide real selectivity.

### Optimize the workload class, not one screenshot

Look at representative plans across:

- hot transactional lookups
- reporting queries
- pagination and feed queries
- background maintenance or batch paths

The goal is stable behavior under realistic data and concurrency, not one perfect explain output.

### Planning and locking interact in production

A query that seems CPU- or IO-heavy in isolation may become much worse when it also extends lock time or interacts with queueing on hot tables.

## Common Failure Modes

### One-time plan success mistaken for safety

Teams optimize one captured plan and ignore whether the query stays healthy as parameters, selectivity, and data distribution shift.

### ORM opacity on hot paths

The application hides SQL shape behind abstraction until a high-value path becomes expensive and difficult to reason about.

### Scan reduction without workflow reduction

One query improves, but the overall business flow still performs too many round trips or redundant reads.

## Principal Review Lens

- Which estimate error is driving the bad plan?
- Is the bottleneck CPU, IO, memory, or lock interaction?
- Will this plan remain good as data distribution shifts?
- Which query is currently “acceptable” only because current scale has not exposed its weakest plan yet?
