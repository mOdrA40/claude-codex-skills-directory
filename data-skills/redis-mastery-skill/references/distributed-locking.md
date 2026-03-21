# Distributed Locking (Redis)

## Rules

- Locks need timeouts, ownership, and failure semantics understood.
- Redis locks are coordination tools, not magic correctness guarantees.
- Prefer idempotency and invariant design over overusing locks.
- Recovery from holder crash must be part of the design.

## Principal Review Lens

- What invariant actually depends on this lock?
- What happens if lock expires during long work?
- Is this solving a design problem with a fragile primitive?
