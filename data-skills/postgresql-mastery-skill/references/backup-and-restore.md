# Backup and Restore (PostgreSQL)

## Rules

- A backup strategy is incomplete without restore drills.
- Define RPO and RTO explicitly.
- Validate object-level and cluster-level recovery paths.
- Protect backup integrity, access control, and retention.

## Recovery Heuristics

### Restore truth matters more than backup confidence

The meaningful question is not whether backups exist, but whether the team knows what can be restored, how long it takes, and what business paths remain impaired during recovery.

### Surgical recovery often matters most

In many incidents the highest-value capability is the ability to restore one table, tenant, schema, or time slice safely rather than thinking only in terms of full-cluster recovery.

### Backup posture includes operational security

Backup retention, repository access, integrity validation, and restore tooling are all part of whether the recovery promise is trustworthy.

## Common Failure Modes

### Backup checkbox confidence

Teams feel safe because scheduled backups succeed, but real restore time and failure handling have not been exercised meaningfully.

### Full-restore imagination only

The team can describe disaster recovery at cluster level but lacks precise procedures for more common partial or surgical restore needs.

### Recovery speed overestimated

Restore timing is assumed from optimistic theory rather than realistic database size, verification, and application reattachment behavior.

## Principal Review Lens

- How long does restore really take for the largest database?
- What data loss window is acceptable?
- Can the team restore one tenant/table/workload surgically if needed?
- Which recovery assumption is currently most weakly proven under realistic pressure?
