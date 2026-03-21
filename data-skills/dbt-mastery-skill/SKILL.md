---
name: dbt-principal-engineer
description: |
  Principal/Senior-level dbt playbook for analytics engineering architecture, model design, testing, lineage, governance, and operating trustworthy transformation platforms at scale.
  Use when: designing dbt projects, reviewing model and domain boundaries, governing analytics transformations, or operating dbt in production workflows.
---

# dbt Mastery (Senior → Principal)

## Operate

- Start from data product boundaries, lineage trust, and transformation ownership.
- Treat dbt as analytics engineering infrastructure, not only SQL templating.
- Prefer clear model layers, documented contracts, and governance over convenience sprawl.
- Optimize for trustworthy semantics, maintainability, and platform supportability.

## Default Standards

- Model boundaries should reflect domain ownership and consumer intent.
- Tests and documentation must support trust, not vanity coverage.
- Macros should reduce repetition without hiding logic.
- Environments and runs should align with deployment and governance policy.
- dbt should enable governed self-service, not analytics chaos.

## References

- Project architecture and model layering: [references/project-architecture-and-model-layering.md](references/project-architecture-and-model-layering.md)
- Testing, contracts, and trust signals: [references/testing-contracts-and-trust-signals.md](references/testing-contracts-and-trust-signals.md)
- Macros, packages, and abstraction discipline: [references/macros-packages-and-abstraction-discipline.md](references/macros-packages-and-abstraction-discipline.md)
- Lineage, ownership, and data product governance: [references/lineage-ownership-and-data-product-governance.md](references/lineage-ownership-and-data-product-governance.md)
- Environments, deployment, and run safety: [references/environments-deployment-and-run-safety.md](references/environments-deployment-and-run-safety.md)
- Warehouse economics and performance discipline: [references/warehouse-economics-and-performance-discipline.md](references/warehouse-economics-and-performance-discipline.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
