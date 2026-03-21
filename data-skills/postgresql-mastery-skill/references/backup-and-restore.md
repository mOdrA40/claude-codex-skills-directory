# Backup and Restore (PostgreSQL)

## Rules

- A backup strategy is incomplete without restore drills.
- Define RPO and RTO explicitly.
- Validate object-level and cluster-level recovery paths.
- Protect backup integrity, access control, and retention.

## Principal Review Lens

- How long does restore really take for the largest database?
- What data loss window is acceptable?
- Can the team restore one tenant/table/workload surgically if needed?
