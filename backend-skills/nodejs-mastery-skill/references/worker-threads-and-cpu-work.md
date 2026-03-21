# Worker Threads and CPU-Bound Work

## Purpose

Node.js request paths should not quietly absorb CPU-heavy work until the event loop degrades. Principal-level design means deciding when CPU work stays in-process, moves to worker threads, or leaves the service entirely.

## Decision Heuristics

Keep work on the main event loop only when:

- it is small and predictable
- latency impact is negligible
- serialization overhead to workers would be worse

Use worker threads when:

- CPU-heavy transforms exist
- image, crypto, compression, parsing, or document conversion is significant
- work must remain local to the process for latency or deployment reasons

Move work out-of-process when:

- tasks are long-running
- failure isolation matters more than locality
- retry/replay semantics matter operationally

## Bad vs Good

```text
❌ BAD
A request path performs large PDF parsing synchronously and p95 latency climbs for the entire service.

✅ GOOD
CPU-heavy work is isolated with bounded concurrency and explicit timeout or queue policy.
```

## Review Questions

- What is the worst-case CPU cost of this request?
- Can one request monopolize the process?
- Is concurrency bounded for worker execution?
- Can cancellation stop useless CPU work?
