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

## Principal Review Lens

- What state change is most likely to break restore safety?
- Can the team recover this job after a failed deployment quickly?
- Are state and checkpoint settings matched to real workload size?
- Which upgrade path most deserves rehearsal?
