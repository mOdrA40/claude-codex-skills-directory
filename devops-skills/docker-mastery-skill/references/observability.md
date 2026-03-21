# Observability (Docker)

## Rules

- Monitor container restarts, OOMs, CPU throttling, and startup latency.
- Container logs should be structured and aggregation-ready.
- Host and container signals must be correlated.
- Alerts should point toward workload or runtime root cause.

## Principal Review Lens

- Which metric predicts a broken rollout earliest?
- Can we distinguish app crash from runtime/resource misconfiguration quickly?
- Are logs and metrics sufficient without attaching shells everywhere?
