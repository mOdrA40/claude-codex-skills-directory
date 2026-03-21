# Observability (Kubernetes)

## Rules

- Observe cluster health, workload health, and user-facing SLOs separately.
- Logs, metrics, events, and traces should complement each other.
- Alerts must tell on-call where to start, not only that YAML exists.
- Cardinality control matters in cluster telemetry.

## Principal Review Lens

- Which signal predicts platform pain earliest?
- Can we isolate cluster issue versus app issue quickly?
- Are alerts actionable under pressure?
