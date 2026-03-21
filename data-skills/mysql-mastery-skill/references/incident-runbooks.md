# Incident Runbooks (MySQL)

## Cover at Minimum

- Replication lag spike.
- Lock storm or deadlock surge.
- Connection saturation.
- Bad migration rollout.
- Backup/restore emergency.
- Failover event.

## Response Rules

- Protect correctness before chasing ideal performance.
- Stabilize top workload and blast radius first.
- Capture evidence around locks, lag, and slow queries before broad tuning changes.
- Communicate clearly about stale reads, write impact, or recovery windows.

## Principal Review Lens

- Can on-call reduce pain in 10 minutes?
- Which emergency action risks data inconsistency most?
- What evidence confirms the system is truly stable again?
- Are runbooks aligned with real workload behavior?
