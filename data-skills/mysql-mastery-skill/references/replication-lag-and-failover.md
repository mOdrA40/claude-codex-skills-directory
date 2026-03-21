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

## Principal Review Lens

- What user-visible behavior changes during failover?
- Which service silently assumes zero lag?
- Can the team explain the topology under incident pressure?
- What hidden dependency would make failover slower than promised?
