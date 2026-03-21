# Incident Runbooks (MySQL)

## Cover at Minimum

- Replication lag spike.
- Lock storm or deadlock surge.
- Connection saturation.
- Bad migration rollout.
- Backup/restore emergency.
- Failover event.

## Incident Heuristics

### Triage by workload path first

Clarify whether the main pain is:

- write-path contention
- read-path saturation or stale read risk
- replication / failover behavior
- operational change side effect

### Protect correctness before throughput optics

It is better to degrade non-critical traffic than to hide a correctness, lag, or failover problem behind temporary throughput relief.

### Recovery must be measurable

Runbooks should define what proves recovery, such as reduced lock waits, restored lag posture, stable failover routing, or normalized slow-query pressure.

## Response Rules

- Protect correctness before chasing ideal performance.
- Stabilize top workload and blast radius first.
- Capture evidence around locks, lag, and slow queries before broad tuning changes.
- Communicate clearly about stale reads, write impact, or recovery windows.

## Common Failure Modes

### Lag calm, application pain

Infrastructure metrics look acceptable again, but application routing or stale-read assumptions still create user-visible inconsistency.

### Emergency action without workload understanding

The team acts quickly, but the chosen intervention worsens the hottest path or creates recovery debt later.

## Principal Review Lens

- Can on-call reduce pain in 10 minutes?
- Which emergency action risks data inconsistency most?
- What evidence confirms the system is truly stable again?
- Are runbooks aligned with real workload behavior?
- Which MySQL incident still lacks a safe-first playbook tied to real workload shape?
