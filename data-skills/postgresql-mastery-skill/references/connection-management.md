# Connection Management (PostgreSQL)

## Rules

- Connections are finite; pool sizing is architecture, not a default.
- Watch queueing, saturation, and idle connection waste.
- Use transaction pooling only when semantics remain safe.
- Know how application concurrency maps to DB concurrency.

## Principal Review Lens

- What fails first when connection demand spikes?
- Are timeouts aligned across app, pooler, and database?
- Is connection count hiding slow query or lock problems?
