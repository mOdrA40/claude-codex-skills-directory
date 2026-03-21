# Query Planning (PostgreSQL)

## Defaults

- Use `EXPLAIN (ANALYZE, BUFFERS)` on real hotspots.
- Optimize for stable query shape, not one lucky plan.
- Review row estimates, join order, scan type, and sort cost together.
- Favor query clarity over ORM-generated surprises on hot paths.

## Principal Review Lens

- Which estimate error is driving the bad plan?
- Is the bottleneck CPU, IO, memory, or lock interaction?
- Will this plan remain good as data distribution shifts?
