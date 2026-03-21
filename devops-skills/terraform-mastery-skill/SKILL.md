---
name: terraform-principal-engineer
description: |
  Principal/Senior-level Terraform playbook for infrastructure modeling, module design, state management, governance, multi-environment delivery, and safe platform change management.
  Use when: designing IaC architecture, reviewing Terraform modules, operating shared state, scaling platform teams, or hardening infrastructure delivery workflows.
---

# Terraform Mastery (Senior → Principal)

## Operate

- Start from ownership boundaries, blast radius, and change safety.
- Treat Terraform as infrastructure lifecycle management, not just file generation.
- Prefer module clarity, predictable plans, and explicit environment strategy.
- Optimize for safe review, repeatable rollout, and recovery from bad state or bad assumptions.

## Default Standards

- Modules should encode reusable intent, not abstract every resource mindlessly.
- State strategy must reflect team boundaries and failure domains.
- Plans should be reviewable by humans.
- Provider/version policy should be deliberate.
- Drift, imports, and emergency changes need known workflows.

## References

- Module design and composition: [references/module-design-and-composition.md](references/module-design-and-composition.md)
- State management and backends: [references/state-management-and-backends.md](references/state-management-and-backends.md)
- Environment strategy and promotion: [references/environment-strategy-and-promotion.md](references/environment-strategy-and-promotion.md)
- Drift detection and lifecycle control: [references/drift-detection-and-lifecycle-control.md](references/drift-detection-and-lifecycle-control.md)
- Reviewability and plan safety: [references/reviewability-and-plan-safety.md](references/reviewability-and-plan-safety.md)
- Governance, policy, and platform teams: [references/governance-policy-and-platform-teams.md](references/governance-policy-and-platform-teams.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
