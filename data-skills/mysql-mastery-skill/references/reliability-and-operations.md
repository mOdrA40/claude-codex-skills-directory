# Reliability and Operations (MySQL)

## Operational Defaults

- Monitor slow queries, connections, locks, deadlocks, replication lag, disk, and backup health.
- Keep upgrades, schema changes, and failover steps rehearsed and reversible.
- Distinguish database symptoms from app-side concurrency or pool misconfiguration.
- Document operator-safe actions for hot incidents.

## Run-the-System Thinking

- Primary and replicas need role-appropriate dashboards and alert thresholds.
- Capacity planning should include spikes, backfills, maintenance, and node loss.
- DBAs and app teams need shared understanding of hot paths and risk.
- Operational simplicity often beats feature maximization.

## Principal Review Lens

- Which metric gives the earliest warning of user pain?
- What action would stabilize the system fastest during a contention storm?
- Can the team explain top workload consumers without guesswork?
- Are we operating a database service or reacting to ORM surprises?
