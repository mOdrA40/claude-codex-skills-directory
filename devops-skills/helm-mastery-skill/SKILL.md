---
name: helm-principal-engineer
description: |
  Principal/Senior-level Helm playbook for chart design, release safety, templating discipline, values governance, upgrade strategy, and Kubernetes application delivery at scale.
  Use when: designing reusable charts, reviewing release workflows, operating multi-team chart ecosystems, or hardening Kubernetes deployment standards.
---

# Helm Mastery (Senior → Principal)

## Operate

- Start from workload ownership, release safety, and chart readability.
- Treat Helm as a packaging and release contract, not as a free-form templating engine.
- Prefer charts that are boring to render, diff, review, and upgrade.
- Optimize for safe defaults, explicit values, and operational trust.

## Default Standards

- Templates should remain readable by humans.
- Values must reflect ownership and safe override boundaries.
- Upgrades and rollbacks need known behavior.
- Chart APIs should evolve deliberately.
- One chart should not become a platform-in-a-file anti-pattern.

## References

- Chart design and boundaries: [references/chart-design-and-boundaries.md](references/chart-design-and-boundaries.md)
- Template readability and functions: [references/template-readability-and-functions.md](references/template-readability-and-functions.md)
- Values strategy and override governance: [references/values-strategy-and-override-governance.md](references/values-strategy-and-override-governance.md)
- Release safety and upgrade behavior: [references/release-safety-and-upgrade-behavior.md](references/release-safety-and-upgrade-behavior.md)
- Testing, linting, and diff workflows: [references/testing-linting-and-diff-workflows.md](references/testing-linting-and-diff-workflows.md)
- Multi-team chart ecosystems: [references/multi-team-chart-ecosystems.md](references/multi-team-chart-ecosystems.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
