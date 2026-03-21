# Observability and Debugging (OPA Gatekeeper)

## Rules

- Policy observability should reveal deny reasons, latency tax, audit drift, and template health clearly.
- Debugging must distinguish template bugs, resource-shape mismatches, and intended enforcement.
- Operators need dashboards that support both platform health and team-facing issue triage.
- Admission-path debugging should be practical under outage pressure.

## Useful Signals

- Deny volume, audit violations, admission latency, webhook health, template errors, and exception usage.
- Correlate policy denials with deploy and cluster-change events.
- Standardize debugging paths for common policy failures.
- Preserve evidence for why a workload was blocked.

## Principal Review Lens

- Can responders explain a denial quickly and accurately?
- Which missing signal causes most policy confusion today?
- Are teams blaming policy for unrelated deployment failures?
- What observability improvement most reduces MTTR?
