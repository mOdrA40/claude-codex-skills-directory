# Disaster Recovery

## Rules

- Federation, shovel, replication, and backups solve different DR problems.
- Define acceptable message loss and replay expectations explicitly.
- Test recovery paths, not just topology diagrams.
- Protect management credentials and exported definitions.

## DR Heuristics

### DR should be described in message semantics, not only topology

The most important question is what messages may be lost, duplicated, delayed, or replayed during disaster recovery—not just which nodes or links exist.

### Different DR tools imply different promises

Federation, shovel, backup/export, and cross-site designs each trade off freshness, durability, operational complexity, and replay control differently.

### Restore clarity matters as much as failover design

The team should know how service returns, how backlog behaves, how tenant boundaries are preserved, and what operators tell application teams during recovery.

## Common Failure Modes

### Diagram confidence without semantic clarity

The DR topology looks sophisticated, but nobody can clearly explain what it means for message loss, replay, or ordering under failure.

### Failover optimism, restore confusion

Traffic can move or infrastructure can come back, but the platform lacks a clear story for backlog reconciliation and downstream semantic safety.

### Credential and definition recovery blind spot

The infrastructure path is considered, but access, exported definitions, and secure restore workflows remain too weakly practiced.

## Principal Review Lens

- What is the practical RPO/RTO for the platform?
- Which messages are safe to lose or replay?
- Can the team restore service without violating tenant boundaries?
- Which RabbitMQ recovery promise is most likely to fail under a real regional or site-loss event?
