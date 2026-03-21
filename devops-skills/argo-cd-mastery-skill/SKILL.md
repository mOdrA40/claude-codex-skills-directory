---
name: argocd-principal-engineer
description: |
  Principal/Senior-level Argo CD playbook for GitOps architecture, application boundaries, sync safety, tenancy, policy, and operating continuous delivery platforms on Kubernetes.
  Use when: designing GitOps delivery, reviewing Argo CD app structure, operating multi-team clusters, or hardening declarative deployment workflows.
---

# Argo CD Mastery (Senior → Principal)

## Operate

- Start from desired-state ownership, environment promotion, and blast radius.
- Treat Argo CD as a delivery control plane, not just a sync robot.
- Prefer clear app boundaries, reviewable diffs, and controlled reconciliation.
- Optimize for safe promotion, operational trust, and debuggable drift handling.

## Default Standards

- Application boundaries must reflect ownership and rollback needs.
- Auto-sync should be deliberate, not default cargo cult.
- Drift policy and emergency change handling must be explicit.
- Multi-tenant app/project governance should be strict enough to protect clusters.
- Sync waves, hooks, and generators should remain understandable.

## References

- GitOps architecture and application boundaries: [references/gitops-architecture-and-application-boundaries.md](references/gitops-architecture-and-application-boundaries.md)
- Sync policy, promotion, and rollout safety: [references/sync-policy-promotion-and-rollout-safety.md](references/sync-policy-promotion-and-rollout-safety.md)
- App-of-apps, generators, and composition tradeoffs: [references/app-of-apps-generators-and-composition-tradeoffs.md](references/app-of-apps-generators-and-composition-tradeoffs.md)
- Multi-tenant governance and project security: [references/multi-tenant-governance-and-project-security.md](references/multi-tenant-governance-and-project-security.md)
- Drift, emergency change, and reconciliation control: [references/drift-emergency-change-and-reconciliation-control.md](references/drift-emergency-change-and-reconciliation-control.md)
- Observability and operational debugging: [references/observability-and-operational-debugging.md](references/observability-and-operational-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
