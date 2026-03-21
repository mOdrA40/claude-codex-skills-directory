# Observability (CockroachDB)

## Rules

- Monitor latency, retries, contention, range health, and regional skew.
- Alert on signals that precede user-visible pain.
- Tie statement behavior to topology and workload class.
- Dashboards should separate healthy retries from dangerous storms.

## Principal Review Lens

- Which metric reveals contention earliest?
- Can we isolate a hot tenant, region, or statement quickly?
- Are alerts about actual risk or just noisy internals?
