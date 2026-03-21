# State, Checkpoints, and Savepoint Safety

## Rules

- Stateful stream processing lives or dies on checkpoint and state discipline.
- Savepoints and upgrades need explicit compatibility planning.
- State size, backend choice, and recovery time are platform concerns.
- Never assume stateful upgrades are routine without testing.

## Practical Guidance

- Monitor checkpoint duration, failure rate, and state growth.
- Align state backend and retention with recovery expectations.
- Test savepoint restore and job evolution workflows before production dependence.
- Document which code and schema changes are state-compatible.

## Safety Heuristics

### State is a contract, not just a runtime detail

Flink state embodies correctness, recovery speed, and upgrade safety. Teams should treat it like a long-lived contract between job versions, operators, and downstream consumers.

### Recovery time must stay legible

As state size grows, checkpoint success alone is not enough. The platform must know how restore time, backfill catch-up, and replay behavior change under realistic failure conditions.

### Savepoint trust must be rehearsed

The existence of savepoints does not mean the upgrade path is safe. Compatibility, operator practice, and rollback clarity still matter.

## Common Failure Modes

### Checkpoint green, recovery weak

The job checkpoints successfully enough that teams feel safe, but full recovery time and upgrade behavior are still poorly understood.

### State growth normalized

State size increases gradually until checkpoint cost, recovery time, and storage pressure create platform fragility.

## Principal Review Lens

- What state change is most likely to break restore safety?
- Can the team recover this job after a failed deployment quickly?
- Are state and checkpoint settings matched to real workload size?
- Which upgrade path most deserves rehearsal?
- What state assumption is currently least proven under real rollback pressure?
