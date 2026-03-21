# Multi-Tenant Governance and Platform Standards

## Rules

- Shared Tekton platforms need clear governance over tasks, images, secrets, and namespaces.
- One tenant should not be able to create unsafe or noisy execution for others.
- Platform teams should publish safe task patterns and trusted execution models.
- Governance should reduce repeated CI/CD risk, not just add process.

## Practical Guidance

- Standardize trusted base images and high-risk task patterns.
- Track owners of shared tasks and bundles.
- Limit who can create privileged service accounts or deployment-capable pipelines.
- Keep exceptions explicit and reviewable.

## Principal Review Lens

- Which tenant has the highest execution blast radius today?
- Are platform standards strong enough to prevent common mistakes?
- What shared task is ownerless or dangerously broad?
- Which governance control most improves safety per unit effort?
