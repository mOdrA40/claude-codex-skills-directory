# Sync Policy, Promotion, and Rollout Safety

## Rules

- Sync policy should reflect environment criticality and change risk.
- Automatic reconciliation is powerful but must be bounded by trust and observability.
- Promotion workflows should make state transitions explicit.
- Rollback behavior matters as much as initial sync success.

## Practical Guidance

- Use manual gates for high-blast-radius or low-confidence changes where needed.
- Make sync windows, health checks, and rollout orchestration understandable.
- Align Git promotion strategy with operational support reality.
- Preserve evidence around failed or partial reconciliations.

## Principal Review Lens

- Which app should not be auto-syncing today?
- Can the team predict what happens during a bad rollout?
- What promotion step is least trustworthy right now?
- Are we automating safety or automating drift into production?
