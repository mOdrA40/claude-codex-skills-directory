# Observability (PostgreSQL)

## Rules

- Monitor latency, locks, deadlocks, checkpoints, replication lag, and connections.
- Slow query visibility should include fingerprints and plan drift clues.
- Align database telemetry with application traces and incidents.
- Alert on symptoms that predict user pain, not vanity counters.

## Observability Heuristics

### Good PostgreSQL observability explains queueing, not just counts

Teams need to know what is blocked, what is waiting, what is slow, and what workload class is actually causing user pain.

### Database signals should connect to application reality

The most useful dashboards help engineers trace app latency, lock storms, migration pain, or replica lag back to the relevant database behavior quickly.

### One table or one query class can matter more than averages

Cluster-wide medians can hide the single hot path, migration, or background job that is causing the incident.

## Common Failure Modes

### Metric abundance without causal clarity

The team can see many counters moving but still cannot explain whether the problem began with locks, bad plans, checkpoints, replication, or application concurrency.

### Alerting on database internals without user translation

The system pages because numbers moved, but operators still do not know what business journey is at risk.

### Slow-query visibility without workflow context

A fingerprint is identified, but teams cannot map it back to the owning service or critical user path quickly enough.

## Principal Review Lens

- Which metric warns earliest before saturation?
- Can we correlate app regressions to DB behavior quickly?
- Are alerts actionable or noisy?
- Which missing signal would most reduce diagnosis time during a real PostgreSQL incident?
