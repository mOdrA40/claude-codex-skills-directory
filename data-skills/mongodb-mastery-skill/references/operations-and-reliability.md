# Operations and Reliability (MongoDB)

## Operational Defaults

- Monitor replication lag, election frequency, and slow queries.
- Treat shard imbalance and hot chunks as production issues, not edge cases.
- Keep backup/restore workflows tested.
- Watch working set vs available memory.

## Reliability Rules

- Know write concern and read concern tradeoffs per workload.
- Design for idempotent retry behavior where clients may retry.
- Validate behavior during primary stepdown.

## Principal Review Lens

- What consistency level does this flow actually need?
- How does the system behave during elections?
- Can one tenant or workload create a hot shard?
