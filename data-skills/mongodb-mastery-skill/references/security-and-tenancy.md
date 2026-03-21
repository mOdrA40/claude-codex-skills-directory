# Security and Tenancy (MongoDB)

## Rules

- Least privilege applies to app users, admin users, and automation.
- Tenant isolation must survive debugging, exports, and restores.
- Protect network paths, backups, and cluster admin surfaces.
- Auditing and masking need explicit design.

## Principal Review Lens

- Which operator or role can see too much data today?
- How is tenant scope enforced and verified?
- What shortcut during incident response would violate boundaries?
