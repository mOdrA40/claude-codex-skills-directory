# Auth Methods, Identity, and Policy Design

## Rules

- Authentication and policy should reflect real identity and least-privilege boundaries.
- Auth method sprawl increases complexity and audit burden.
- Human and machine access patterns should be separated deliberately.
- Policies should remain understandable and reviewable.

## Practical Guidance

- Standardize auth methods where platform maturity allows.
- Keep role/policy naming tied to ownership and purpose.
- Review wildcard or path-based grants carefully.
- Align onboarding and offboarding workflows with identity sources.

## Principal Review Lens

- Which auth method has the weakest operational discipline?
- What policy grant is broader than its owner realizes?
- Can reviewers understand what access a role truly has?
- What auth simplification most reduces risk?
