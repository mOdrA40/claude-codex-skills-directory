# Go Workers and Queue Consumers

## Purpose

Many Go backends are not just HTTP servers. They are also workers, consumers, cron processors, and event handlers. Those paths need as much rigor as APIs.

## Rules

- every goroutine has an owner
- consumption concurrency is bounded
- shutdown drains or stops intentionally
- idempotency is explicit
- poison-message behavior is defined
- dependency timeouts exist for every side effect

## Bad vs Good

```text
❌ BAD
A consumer spawns goroutines freely for every message and shutdown just kills the process.

✅ GOOD
The consumer owns a bounded worker pool, tracks in-flight work, and drains with deadlines on shutdown.
```

## Review Questions

- what happens when one message takes 10x longer than expected?
- can duplicates happen and are they safe?
- is retry ownership in the broker, consumer, or both?
- what metrics show lag, age, and failure class?
