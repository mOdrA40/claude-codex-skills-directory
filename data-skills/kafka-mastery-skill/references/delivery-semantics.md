# Delivery Semantics (Kafka)

## Rules

- At-least-once is common; exactly-once claims need careful boundary review.
- Consumers must stay idempotent even when brokers behave correctly.
- Retry and DLQ policy should be designed around business impact.
- Duplicate and out-of-order handling belongs in system design.

## Semantics Heuristics

### Delivery guarantees stop at system boundaries

Kafka can provide strong guarantees within its own mechanisms, but once workflows involve external APIs, databases, notifications, or side effects, the true delivery story depends on application design.

### Exactly-once language must be scoped precisely

Teams should be explicit about whether they mean:

- broker-level semantics
- stream-processing semantics
- end-to-end business effect semantics

Confusing those levels is a common source of overconfidence.

### Idempotency is the practical safety net

Even when broker and client behavior are healthy, retries, rebalances, and replays still make idempotent consumers the safer default posture.

## Common Failure Modes

### Exactly-once theater

The architecture claims exactly-once behavior broadly, but important downstream effects are still exposed to duplicates or ambiguous replay.

### Broker confidence, workflow ambiguity

Kafka behaves correctly while the surrounding service boundaries remain semantically unsafe on retry or replay.

### Ordering assumptions leaking across partitions

Teams implicitly rely on stronger ordering than Kafka actually guarantees for the chosen partition model.

## Principal Review Lens

- Where can duplicates still happen despite Kafka features?
- What external side effects are not transactionally protected?
- Which consumer assumption breaks first under replay?
- What semantic promise are we implying to the business that the current system does not fully keep?
