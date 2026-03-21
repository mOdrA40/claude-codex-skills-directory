# Replication and Failover (PostgreSQL)

## Rules

- Replication is not backup.
- Know acceptable lag and read-staleness by workload.
- Test failover with real applications, not only infra scripts.
- Reconnection and retry behavior must be observable.

## Principal Review Lens

- What user-visible behavior changes during failover?
- Is read traffic safe on replicas for this workload?
- How quickly can the team detect split-brain assumptions or stale reads?
