# Queue Poison Message and Dead-Letter Playbook for Node.js Services

## Purpose

Not every failed message should be retried forever. Some messages are toxic, malformed, or repeatedly trigger non-transient failure.

## Rules

- classify transient vs terminal failure
- bound retries explicitly
- surface dead-letter count, age, and cause
- make replay an operator-safe action
- keep poison traffic isolated from healthy traffic

## Agent Questions

- what makes this message retryable?
- when should it go to dead-letter?
- how can operators replay safely without duplicating side effects?
- can one poison message block a shared partition or worker?

## Principal Heuristics

- Dead-letter queues are for controlled failure, not silent abandonment.
- If replay is unsafe, message handling is under-designed.
