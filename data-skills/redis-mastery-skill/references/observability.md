# Observability (Redis)

## Rules

- Monitor latency, hit ratio, evictions, memory, fragmentation, and replication state.
- Distinguish healthy misses from cache collapse.
- Big-key and hot-key detection should be routine.
- Correlate Redis pain with downstream database load.

## Principal Review Lens

- Which metric predicts cascading failure earliest?
- Can we identify hot tenants or hot features quickly?
- Are dashboards telling us correctness risk or just speed?
