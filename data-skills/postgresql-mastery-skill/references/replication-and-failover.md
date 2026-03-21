# Replication and Failover (PostgreSQL)

## Rules

- Replication is not backup.
- Know acceptable lag and read-staleness by workload.
- Test failover with real applications, not only infra scripts.
- Reconnection and retry behavior must be observable.

## Replication Heuristics

### Define what replicas are trusted to do

Replica usage should be explicit about:

- read staleness tolerance
- consistency expectations
- reporting vs user-facing workload suitability
- behavior during lag spikes and role transitions

### Failover is an application event, not just an infra event

The cluster may fail over successfully while the application still misbehaves due to stale reads, reconnect storms, or wrong assumptions about write availability.

## Common Failure Modes

### Healthy replica metrics, unsafe application reads

The infrastructure looks fine, but the workload depends on fresher state than the replica path can safely provide.

### Failover script success, user-visible failure

Infrastructure automation completes, but connection pools, read/write routing, or retry posture create a second incident in the application layer.

### Lag normalized as background noise

Replication lag becomes culturally normal until one incident proves the business path depended on fresher data than assumed.

## Principal Review Lens

- What user-visible behavior changes during failover?
- Is read traffic safe on replicas for this workload?
- How quickly can the team detect split-brain assumptions or stale reads?
- Which failover assumption is currently untested under real application load?
