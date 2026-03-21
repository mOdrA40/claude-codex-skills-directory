# Observability (MongoDB)

## Rules

- Monitor slow queries, lag, elections, cache efficiency, and page faults.
- Dashboards should distinguish healthy churn from dangerous instability.
- Query regressions need visibility into planner and index usage.
- Correlate database telemetry with application latency and errors.

## Principal Review Lens

- Which metric predicts customer pain earliest?
- Can we isolate hot collections or hot tenants quickly?
- Are alerts about actual risk or background noise?
