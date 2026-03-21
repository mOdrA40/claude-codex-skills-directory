# Disaster Recovery

## Rules

- Federation, shovel, replication, and backups solve different DR problems.
- Define acceptable message loss and replay expectations explicitly.
- Test recovery paths, not just topology diagrams.
- Protect management credentials and exported definitions.

## Principal Review Lens

- What is the practical RPO/RTO for the platform?
- Which messages are safe to lose or replay?
- Can the team restore service without violating tenant boundaries?
