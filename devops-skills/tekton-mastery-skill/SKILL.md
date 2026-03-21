---
name: tekton-principal-engineer
description: |
  Principal/Senior-level Tekton playbook for Kubernetes-native CI/CD pipelines, task design, execution isolation, supply-chain integrity, and platform-scale pipeline operations.
  Use when: designing pipeline platforms on Kubernetes, reviewing Tekton tasks, securing build flows, or operating multi-tenant CI/CD systems.
---

# Tekton Mastery (Senior → Principal)

## Operate

- Start from trust boundaries, execution isolation, and supply-chain requirements.
- Treat Tekton as a platform for workload execution, not just YAML for builds.
- Prefer composable tasks, clear pipeline boundaries, and auditable execution.
- Optimize for secure builds, predictable delivery, and multi-tenant supportability.

## Default Standards

- Task design should be composable and reviewable.
- Workspaces, params, and results should reflect explicit contracts.
- Build isolation and secret handling must be strong.
- Supply-chain provenance should be intentional.
- Platform operators need clear insight into failed or unsafe pipeline behavior.

## References

- Pipeline architecture and task boundaries: [references/pipeline-architecture-and-task-boundaries.md](references/pipeline-architecture-and-task-boundaries.md)
- Workspaces, params, results, and contract design: [references/workspaces-params-results-and-contract-design.md](references/workspaces-params-results-and-contract-design.md)
- Runner isolation, build security, and secret handling: [references/runner-isolation-build-security-and-secret-handling.md](references/runner-isolation-build-security-and-secret-handling.md)
- Supply-chain integrity and provenance: [references/supply-chain-integrity-and-provenance.md](references/supply-chain-integrity-and-provenance.md)
- Multi-tenant governance and platform standards: [references/multi-tenant-governance-and-platform-standards.md](references/multi-tenant-governance-and-platform-standards.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
