# Observability (PostgreSQL)

## Rules

- Monitor latency, locks, deadlocks, checkpoints, replication lag, and connections.
- Slow query visibility should include fingerprints and plan drift clues.
- Align database telemetry with application traces and incidents.
- Alert on symptoms that predict user pain, not vanity counters.

## Principal Review Lens

- Which metric warns earliest before saturation?
- Can we correlate app regressions to DB behavior quickly?
- Are alerts actionable or noisy?
