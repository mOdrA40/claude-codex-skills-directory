---
name: opa-gatekeeper-principal-engineer
description: |
  Principal/Senior-level OPA Gatekeeper playbook for policy architecture, constraint design, admission control safety, multi-tenant governance, and operating policy enforcement on Kubernetes at scale.
  Use when: designing cluster policy, reviewing constraints, hardening platform governance, or operating Gatekeeper in multi-team environments.
---

# OPA Gatekeeper Mastery (Senior → Principal)

## Operate

- Start from platform risk, tenant boundaries, and enforcement blast radius.
- Treat Gatekeeper as a policy control plane for Kubernetes, not a place to dump random rules.
- Prefer high-value, explainable constraints over policy sprawl.
- Optimize for safe enforcement, clear exceptions, and debuggable admission behavior.

## Default Standards

- Constraints should target real risk classes.
- Rego and templates must remain readable to humans.
- Audit and admission behavior should be designed together.
- Exemptions should be explicit and reviewable.
- Multi-cluster and multi-tenant policy governance must be intentional.

## References

- Policy architecture and risk boundaries: [references/policy-architecture-and-risk-boundaries.md](references/policy-architecture-and-risk-boundaries.md)
- Constraint templates and Rego design: [references/constraint-templates-and-rego-design.md](references/constraint-templates-and-rego-design.md)
- Admission safety and rollout strategy: [references/admission-safety-and-rollout-strategy.md](references/admission-safety-and-rollout-strategy.md)
- Audit mode, drift, and exception management: [references/audit-mode-drift-and-exception-management.md](references/audit-mode-drift-and-exception-management.md)
- Multi-tenant governance and policy ownership: [references/multi-tenant-governance-and-policy-ownership.md](references/multi-tenant-governance-and-policy-ownership.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
