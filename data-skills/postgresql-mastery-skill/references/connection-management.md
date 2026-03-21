# Connection Management (PostgreSQL)

## Rules

- Connections are finite; pool sizing is architecture, not a default.
- Watch queueing, saturation, and idle connection waste.
- Use transaction pooling only when semantics remain safe.
- Know how application concurrency maps to DB concurrency.

## Connection Heuristics

### Connection count is a symptom carrier

PostgreSQL connection problems often reveal deeper issues in app concurrency, slow queries, lock waits, timeout mismatch, or transaction design rather than a simple need for more connections.

### Pooling should reflect workflow behavior

The correct pool and timeout posture depends on how long transactions live, how bursty requests are, and whether the app holds resources across slow external work.

### Queueing is often more useful than raw counts

The most important question is not only how many connections exist, but what work is waiting, what is blocked, and which resource interaction is making concurrency unsafe.

## Common Failure Modes

### Bigger pool reflex

The team increases pool size in response to saturation without fixing the slower or blocked work that made the pool unhealthy in the first place.

### Timeout mismatch

Application, pooler, proxy, and database all use different expectations, creating confusing failure behavior during pressure.

### Transaction pooling semantics surprise

Pooling improves efficiency technically, but the chosen mode conflicts with application assumptions in ways that are only discovered during stress.

## Principal Review Lens

- What fails first when connection demand spikes?
- Are timeouts aligned across app, pooler, and database?
- Is connection count hiding slow query or lock problems?
- Which concurrency assumption is most likely to collapse during a traffic or dependency spike?
