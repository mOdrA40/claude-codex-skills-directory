# Delivery Semantics (Kafka)

## Rules

- At-least-once is common; exactly-once claims need careful boundary review.
- Consumers must stay idempotent even when brokers behave correctly.
- Retry and DLQ policy should be designed around business impact.
- Duplicate and out-of-order handling belongs in system design.

## Principal Review Lens

- Where can duplicates still happen despite Kafka features?
- What external side effects are not transactionally protected?
- Which consumer assumption breaks first under replay?
