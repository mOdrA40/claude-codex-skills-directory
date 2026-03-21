---
name: ansible-principal-engineer
description: |
  Principal/Senior-level Ansible playbook for automation architecture, inventory strategy, role design, change safety, secrets handling, and operating configuration management at scale.
  Use when: designing automation standards, reviewing playbooks and roles, managing fleet configuration, or operating Ansible in production workflows.
---

# Ansible Mastery (Senior → Principal)

## Operate

- Start from ownership, blast radius, and idempotent change safety.
- Treat Ansible as an automation and operations contract, not a shell script launcher.
- Prefer readable roles, predictable inventories, and explicit change boundaries.
- Optimize for safe execution, auditability, and recovery from bad automation.

## Default Standards

- Idempotency is non-negotiable.
- Inventory design must reflect ownership and environment reality.
- Roles should encode reusable operational intent, not random task piles.
- Secrets and privilege escalation require strict discipline.
- Dry runs, diff visibility, and rollback posture matter.

## References

- Automation architecture and execution boundaries: [references/automation-architecture-and-execution-boundaries.md](references/automation-architecture-and-execution-boundaries.md)
- Inventory strategy and environment design: [references/inventory-strategy-and-environment-design.md](references/inventory-strategy-and-environment-design.md)
- Role design and idempotent playbooks: [references/role-design-and-idempotent-playbooks.md](references/role-design-and-idempotent-playbooks.md)
- Secrets, privilege, and security posture: [references/secrets-privilege-and-security-posture.md](references/secrets-privilege-and-security-posture.md)
- Change safety, review, and rollout workflows: [references/change-safety-review-and-rollout-workflows.md](references/change-safety-review-and-rollout-workflows.md)
- Multi-team automation governance: [references/multi-team-automation-governance.md](references/multi-team-automation-governance.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
