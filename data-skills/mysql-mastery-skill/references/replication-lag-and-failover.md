# Replication, Lag, and Failover

## Rules

- Replication topology should follow read patterns, durability goals, and recovery expectations.
- Replica lag is an application correctness concern when reads move off primary.
- Failover must be rehearsed with real clients and connection behavior.
- Replication is not backup.

## Practical Guidance

- Define what stale reads are acceptable, where, and for how long.
- Observe lag, applier health, failover time, and reconnect behavior.
- Distinguish routine read scaling from disaster-recovery architecture.
- Keep operator procedures for promotion and reparenting explicit.

## Replication Heuristics

### Read scaling and failover are different promises

A replica topology that is fine for low-risk read scaling may still be a weak failover posture for business-critical write paths or consistency-sensitive reads.

### Lag is a user-facing truth problem

Replica lag is not just an infrastructure number. It changes what users, services, and downstream jobs are allowed to believe about the current state of the world.

### Recovery must include client behavior

A failover sequence is incomplete if applications, pools, retries, or read/write routing still behave as if the old topology exists.

## Common Failure Modes

### Replica confidence without semantics clarity

Teams point reads at replicas before making it explicit which user journeys can tolerate staleness and which cannot.

### Failover success, application confusion

The database promotes correctly, but clients, caches, or routing layers continue producing stale or failed behavior long after infra says recovery succeeded.

### Lag normalized until it matters

Replica delay becomes routine background noise until one business-critical flow proves the assumption was unsafe.

## Principal Review Lens

- What user-visible behavior changes during failover?
- Which service silently assumes zero lag?
- Can the team explain the topology under incident pressure?
- What hidden dependency would make failover slower than promised?
- Which replica usage today is operationally convenient but semantically risky?
