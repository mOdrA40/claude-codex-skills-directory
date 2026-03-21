# Incident Runbooks (Elasticsearch)

## Rules

- Cover red cluster, yellow cluster, heap pressure, indexing backlog, and slow search incidents.
- Stabilize critical workloads before ideal tuning.
- Include safe and unsafe actions explicitly.
- Recovery should be verified with metrics and user-facing checks.

## Principal Review Lens

- Can on-call reduce user pain in 10 minutes?
- Which emergency action creates worse recovery later?
- What confirms true recovery instead of temporary relief?
