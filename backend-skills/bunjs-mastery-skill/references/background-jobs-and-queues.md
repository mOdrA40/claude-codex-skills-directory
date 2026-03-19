# Background Jobs and Queues (Bun Services)

Background work is where hidden reliability debt accumulates: duplicate side effects, lost jobs, silent retries, and shutdown corruption.

## When to Use Background Jobs

Use jobs when:

- work is slow or bursty,
- user response should not wait,
- retry semantics are required,
- side effects need isolation from request latency.

Do not use jobs just to hide bad request-path design.

## Defaults

- Every job needs an owner, retry policy, timeout, and dead-letter strategy.
- Job payloads should be minimal and versionable.
- Keep handlers idempotent.
- Emit metrics for queue depth, attempts, failures, and processing time.

## Retry Rules

- Retry only transient failures.
- Add jitter and cap attempts.
- Distinguish poison jobs from temporary dependency failure.
- Do not retry forever.

## Shutdown Rules

- Stop taking new jobs.
- Finish or safely requeue in-flight jobs.
- Record partial failure clearly.
- Keep job visibility and lease semantics explicit.

## Principal Review Lens

Ask:

- What prevents duplicate side effects?
- What happens when the worker crashes after partial success?
- How is queue lag detected before customer impact?
- Which jobs are safe to drop, and which require guaranteed recovery?
