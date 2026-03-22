# Dependency Timeouts and Abort Strategy in Bun Services

## Purpose

Fast runtimes do not protect you from slow dependencies. In Bun services, failure to define timeout and abort behavior clearly leads to hanging requests, retry storms, pool starvation, and confusing incidents.

## Core Principle

Every dependency call must have an explicit budget.

That budget should align with:

- end-to-end request timeout
- retry posture
- idempotency of the operation
- user-facing latency expectations
- shutdown behavior

## Timeout Budgeting

Avoid choosing timeouts in isolation.

Example thinking:

- client-facing SLO: 2 seconds
- internal handler budget: 1.6 seconds
- outbound dependency budget: 600 ms
- retry only if the operation is safe and a second attempt still fits the budget

If retries exceed the caller's real deadline, they are not resilience. They are extra load.

## Abort Rules

- pass abort signals to outbound HTTP calls
- cancel expensive downstream work when the request deadline is exceeded
- avoid detached promises that ignore caller cancellation
- stop new dependency calls during shutdown or readiness failure

## Bad vs Good

```typescript
// ❌ BAD: no timeout and no cancellation path.
const response = await fetch(url)
```

```typescript
// ✅ GOOD: deadline-aware dependency call.
const response = await fetch(url, {
  signal: AbortSignal.timeout(800),
})
```

## Retry Discipline

Retries should be:

- bounded
- jittered
- limited to safe operations
- disabled for clear terminal failures
- visible in metrics and logs

Do not retry in handlers, shared clients, and queue consumers simultaneously.

## Review Questions

- what is the total time budget for this request or job?
- does the dependency timeout fit inside that budget?
- what happens when the caller disconnects or shutdown begins?
- can retries create duplicate side effects?
- can operators see timeout rate separately from other errors?

## Principal Heuristics

- Prefer fewer, well-reasoned timeouts over many arbitrary ones.
- Cancellation is part of correctness, not just cleanup.
- If a dependency call can outlive the business value of the request, the design is wrong.
