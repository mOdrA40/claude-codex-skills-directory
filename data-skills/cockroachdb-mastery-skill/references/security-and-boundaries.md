# Security and Boundaries (CockroachDB)

## Rules

- Least privilege applies to SQL users, admin access, and operational tooling.
- Protect cross-region traffic, backups, and admin surfaces.
- Residency and compliance requirements must shape placement decisions.
- Tenant and service boundaries should survive incident shortcuts.

## Principal Review Lens

- Which operator path has excessive blast radius?
- Are compliance and placement goals aligned or in tension?
- What boundary is most likely to erode under operational stress?
