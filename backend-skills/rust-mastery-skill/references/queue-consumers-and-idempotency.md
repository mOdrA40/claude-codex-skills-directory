# Queue Consumers and Idempotency in Rust

## Purpose

Rust services often get the correctness of memory and types right while still missing replay safety, retry discipline, and queue consumer operability.

## Rules

- classify retriable vs terminal failures
- bound consumer concurrency
- make idempotency explicit for side effects
- separate parsing failure from business rejection
- expose lag, message age, retry, and dead-letter signals

## Bad vs Good

```text
❌ BAD
Every failure is retried the same way and duplicate effects are discovered in production.

✅ GOOD
The consumer distinguishes invalid input, dependency timeout, conflict, and terminal poison-message behavior.
```

## Review Questions

- can this command run twice safely?
- where is deduplication or idempotency stored?
- does shutdown interrupt side effects halfway through?
- what does the dead-letter path mean operationally?
