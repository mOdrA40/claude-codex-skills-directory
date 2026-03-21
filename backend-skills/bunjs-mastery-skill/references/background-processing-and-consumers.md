# Background Processing and Consumers in Bun

## Principle

Background jobs and consumers are where hidden runtime assumptions become production bugs. They need explicit ownership, concurrency limits, retry policy, and shutdown behavior.

## Rules

- define owner and stop condition for every consumer
- bound concurrency and queue depth
- classify transient vs terminal failure
- define idempotency for side-effecting jobs
- expose lag, age, retry, and dead-letter signals

## Review Questions

- how does the worker behave during deploy shutdown?
- what happens to in-flight jobs on crash?
- can a poison message stall the consumer loop?
