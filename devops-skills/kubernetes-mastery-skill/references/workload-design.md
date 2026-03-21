# Workload Design (Kubernetes)

## Rules

- Model workloads around SLOs, ownership, and failure boundaries.
- Avoid platform abstractions that hide simple workload needs.
- Manifests should make runtime intent obvious.
- Workload design includes rollback and observability posture.

## Principal Review Lens

- Is this workload shape aligned with business criticality?
- What operational assumption is hidden in the manifest?
- Can another team safely operate this workload?
