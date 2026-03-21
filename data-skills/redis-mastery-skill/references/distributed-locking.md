# Distributed Locking (Redis)

## Rules

- Locks need timeouts, ownership, and failure semantics understood.
- Redis locks are coordination tools, not magic correctness guarantees.
- Prefer idempotency and invariant design over overusing locks.
- Recovery from holder crash must be part of the design.

## Locking Heuristics

### Locks should protect one clear invariant

If a lock exists mainly to hide weak workflow design, missing idempotency, or unclear ownership, it will usually create operational fragility instead of reliable correctness.

### Expiry is part of the semantic contract

A lock timeout is not just a cleanup setting. It defines what happens when work runs long, the holder crashes, or ownership becomes ambiguous.

### Coordination needs recovery thinking

The design should explain what happens when the lock is lost, duplicated, expired too early, or held by a worker that no longer makes forward progress.

## Common Failure Modes

### Lock as architecture substitute

Teams introduce Redis locks to patch over missing workflow boundaries or idempotency, then discover they added a fragile coordination dependency.

### Timeout confidence without workload realism

The chosen expiry looks fine until one slow path, pause, or downstream dependency breaks the assumption badly.

### Ownership ambiguity under recovery

When a holder crashes or a timeout expires, the system cannot clearly explain which worker should act next or what duplicated work means.

## Principal Review Lens

- What invariant actually depends on this lock?
- What happens if lock expires during long work?
- Is this solving a design problem with a fragile primitive?
- Which failure mode of this lock is currently least understood by the team?
