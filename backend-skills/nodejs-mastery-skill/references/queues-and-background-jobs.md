# Queues and Background Jobs

## Principle

Do not hide expensive or failure-prone work behind ad-hoc async functions. If work matters operationally, it needs ownership, observability, and replay semantics.

## Rules

- define idempotency for retried jobs
- bound worker concurrency
- separate critical and best-effort queues
- expose queue depth and job age
- avoid request-path coupling to slow background systems

## Failure Questions

- what happens when the queue is down?
- what happens when the consumer is slow?
- can a job run twice safely?
- how is poison-message behavior handled?
