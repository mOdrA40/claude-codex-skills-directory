---
name: kubernetes-principal-engineer
description: |
  Principal/Senior-level Kubernetes playbook for workload design, cluster operations, networking, security, observability, deployment safety, and platform reliability.
  Use when: designing manifests and controllers, operating clusters, debugging production incidents, reviewing platform architecture, or shipping workloads safely on Kubernetes.
---

# Kubernetes Mastery (Senior → Principal)

## Operate

- Start from workload SLOs, failure model, team ownership, and platform constraints.
- Treat Kubernetes as a distributed control plane, not merely a YAML bucket.
- Prefer operational clarity over abstract platform cleverness.
- Design for rollout safety, debuggability, and failure containment.

## Default Standards

- Requests/limits should be intentional.
- Health probes must reflect real readiness.
- Rollouts need safe defaults and rollback posture.
- RBAC and network policy should be least-privilege by default.
- Observability and runbooks are part of workload design.

## References

- Workload design: [references/workload-design.md](references/workload-design.md)
- Pods, deployments, and rollout safety: [references/pods-deployments-and-rollouts.md](references/pods-deployments-and-rollouts.md)
- Services and ingress: [references/services-and-ingress.md](references/services-and-ingress.md)
- Networking and network policy: [references/networking-and-network-policy.md](references/networking-and-network-policy.md)
- Storage and stateful workloads: [references/storage-and-stateful-workloads.md](references/storage-and-stateful-workloads.md)
- Security and RBAC: [references/security-and-rbac.md](references/security-and-rbac.md)
- Resource management: [references/resource-management.md](references/resource-management.md)
- Scheduling and autoscaling: [references/scheduling-and-autoscaling.md](references/scheduling-and-autoscaling.md)
- Observability: [references/observability.md](references/observability.md)
- Debugging and troubleshooting: [references/debugging-and-troubleshooting.md](references/debugging-and-troubleshooting.md)
- Multi-tenant clusters: [references/multi-tenant-clusters.md](references/multi-tenant-clusters.md)
- Platform operations: [references/platform-operations.md](references/platform-operations.md)
- Disaster recovery: [references/disaster-recovery.md](references/disaster-recovery.md)
- GitOps and delivery workflows: [references/gitops-and-delivery-workflows.md](references/gitops-and-delivery-workflows.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
