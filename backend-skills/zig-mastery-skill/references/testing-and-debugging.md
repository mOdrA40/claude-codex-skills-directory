# Zig Backend Testing and Debugging

## Testing Strategy

Use layers of tests:

- domain tests for invariants
- adapter tests for storage or network integration
- transport tests for protocol handling
- smoke tests for boot and shutdown behavior

## What to Test First

- input validation limits
- timeout handling
- partial failure paths
- shutdown/drain behavior
- memory ownership assumptions on hot paths

## Bad vs Good: Missing Failure Tests

```text
❌ BAD
Only the happy path is tested for a payment or user creation flow.

✅ GOOD
Tests cover dependency timeout, malformed input, duplicate requests, and shutdown during in-flight work.
```

## Debugging Questions

- Is the bug caused by allocator lifetime?
- Is a retry causing duplicate effects?
- Is the slow path dominated by parsing, IO, or allocation churn?
- Is cleanup skipped on error?
- Is the service overloaded or just blocked on one dependency?

## Operational Debugging

During incidents, you want evidence for:

- request class causing errors
- dependency responsible for latency increase
- whether memory pressure is local or systemic
- whether queueing is bounded or exploding

## Review Checklist

- Failure paths are tested.
- Cleanup paths are tested.
- Request limits are tested.
- Shutdown behavior is tested.
- Hot path benchmarks exist before performance claims are accepted.
