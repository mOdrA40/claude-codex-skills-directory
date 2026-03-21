# Publisher Confirms and Delivery

## Rules

- Publisher confirms are baseline for reliable publishing.
- Mandatory routing and unroutable handling should be explicit.
- Message durability and queue durability must align.
- Publish-side idempotency still matters.

## Delivery Heuristics

### A successful publish still needs interpretation

Teams should know exactly what a confirm means in their topology and what it does not guarantee about end-to-end consumer effect, routing safety, or duplicate suppression.

### Routing failures should be first-class signals

Unroutable messages are not edge cases. They are often the earliest sign of topology drift, rollout mismatch, or configuration debt.

### Producer idempotency remains important

Confirms improve reliability, but publish retries, connection interruptions, and application behavior can still produce duplicates if producer design is weak.

## Common Failure Modes

### Confirm confidence beyond scope

The system interprets broker confirmation as if the business workflow is now durably and uniquely completed.

### Unroutable message blindness

Routing issues exist, but operational handling is weak enough that they become silent data-loss or hidden backlog problems.

### Durability mismatch

Messages are marked durable in one place while the surrounding queue or topology choices still make the real promise weaker than operators assume.

## Principal Review Lens

- What does a successful publish actually guarantee here?
- How are unroutable messages surfaced and handled?
- Where can duplicates still appear despite confirms?
- What delivery promise are publishers currently assuming that the topology does not really provide?
