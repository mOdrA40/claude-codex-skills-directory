# Backup, Restore, and DR (CockroachDB)

## Rules

- Distributed backups still require tested restore paths.
- Define regional disaster expectations explicitly.
- Validate restore time and application reconnect behavior.
- Keep backup security, retention, and integrity visible.

## Principal Review Lens

- Can the team restore under regional impairment?
- What is the practical RTO and RPO?
- Which data set or tenant is hardest to recover safely?
