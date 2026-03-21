# Application Patterns (CockroachDB)

## Rules

- Applications must treat retries and idempotency as normal, not exceptional.
- Keep transaction scope narrow around invariants.
- Avoid ORM patterns that create chatty distributed transactions.
- Tie business semantics to consistency needs explicitly.

## Principal Review Lens

- Where will application retry logic duplicate side effects?
- Which pattern creates hidden cross-region chatter?
- Is consistency stronger than the product actually needs?
