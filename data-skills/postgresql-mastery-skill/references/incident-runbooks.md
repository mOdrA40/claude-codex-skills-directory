# Incident Runbooks (PostgreSQL)

## Rules

- Runbooks should start from symptoms: saturation, lag, lock storm, disk pressure, failover.
- Include safe first actions, not only ultimate fixes.
- Protect data integrity before chasing vanity recovery speed.
- Capture rollback and escalation conditions.

## Incident Classes

Useful PostgreSQL runbooks should distinguish between:

- lock storms and blocked sessions
- replication lag or failover instability
- disk, autovacuum, or bloat-related pressure
- slow-query or workload-shape regression
- migration or DDL side effects under live traffic

## Operator Heuristics

### Stabilize user pain first

Initial questions should identify:

- which workload or table is hottest
- whether the issue is read-path, write-path, or replication-path specific
- whether an operational change, release, or migration preceded the incident

### Make safe-first actions explicit

Runbooks should spell out which actions are safe for on-call and which require deeper senior review.

### Recovery must be proven

Recovery should be tied to concrete indicators like queueing relief, lag recovery, lock reduction, or restored latency—not only a temporary drop in alert noise.

## Common Failure Modes

### Generic SQL runbook thinking

The team has a generic database playbook, but it does not reflect PostgreSQL-specific lock, vacuum, replication, or migration behavior.

### Fast-looking partial recovery

One symptom improves while the underlying queueing, lag, or workload-shape problem remains active.

## Principal Review Lens

- Can an on-call engineer stabilize the system in 10 minutes?
- What action is explicitly forbidden during the incident?
- Which metric confirms recovery versus temporary relief?
- Which PostgreSQL incident still lacks a safe-first operator path?
