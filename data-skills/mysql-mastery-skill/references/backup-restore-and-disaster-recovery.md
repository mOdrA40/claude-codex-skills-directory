# Backup, Restore, and Disaster Recovery

## Rules

- Backup strategy is incomplete without restore drills.
- RPO and RTO should be explicit and tested.
- Validate both full-cluster and selective recovery needs.
- Backups need access control, integrity checks, and retention policy.

## Recovery Thinking

- Measure restore time for the largest realistic dataset.
- Know whether tenant-level or table-level recovery is possible and safe.
- Separate disaster recovery design from routine high availability assumptions.
- Keep recovery playbooks aligned with replication and application topology.

## Principal Review Lens

- How long does restore actually take, not ideally?
- What data loss window is the business really accepting?
- Which recovery scenario is hardest to execute correctly?
- Are backups trustworthy enough to rely on during a major incident?
