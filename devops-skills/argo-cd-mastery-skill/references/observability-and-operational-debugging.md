# Observability and Operational Debugging (Argo CD)

## Rules

- Observability should reveal sync state, drift, permission failures, and rollout behavior clearly.
- Distinguish repo problems, render problems, cluster apply problems, and runtime health issues quickly.
- Dashboards should support both platform and app-team workflows.
- GitOps debugging must not require reading internal controller behavior only.

## Useful Signals

- Sync errors, drift counts, render failures, permission denials, reconciliation latency, and app health state.
- Correlate with repo changes, cluster events, and deployment telemetry.
- Make common failure patterns obvious in standard dashboards.
- Preserve audit trails for who changed desired state and who overrode it.

## Principal Review Lens

- Can on-call localize the failure layer quickly?
- Which missing signal causes most GitOps confusion today?
- Are teams blaming Argo for cluster or repo issues too often?
- What debugging workflow most needs simplification?
