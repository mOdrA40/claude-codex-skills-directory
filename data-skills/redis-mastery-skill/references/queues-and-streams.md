# Queues and Streams (Redis)

## Rules

- Queues in Redis are fine when semantics and durability limits are explicit.
- Streams add structure but still require consumer ownership and replay thinking.
- Poison message and duplicate handling must be designed.
- Do not let convenience hide delivery guarantees.

## Principal Review Lens

- Why Redis queue/stream over Kafka or RabbitMQ here?
- What failure loses or duplicates work?
- Who owns replay, retention, and dead-letter behavior?
