# Streaming and Backpressure in Rust Services

## Purpose

Rust can model backpressure well, but that advantage disappears if streams are treated as just another async abstraction without operational limits.

## Core Principle

Backpressure is not an implementation detail. It is the policy that decides whether the system slows safely, rejects work, or collapses under buffered load.

## Rules

- bound stream buffers intentionally
- propagate cancellation on disconnect or timeout
- distinguish producer slowness from consumer slowness
- avoid silent memory growth from buffered streams or channels

## Failure Modes

Watch for:
- channel growth with no hard cap
- slow consumers hiding behind async buffering
- retries reintroducing pressure faster than downstream recovers
- per-tenant or per-stream starvation

## Bad vs Good

```text
❌ BAD
Buffered channels absorb load until memory and latency explode.

✅ GOOD
Buffers are bounded, cancellation propagates, and overload becomes visible before the service degrades catastrophically.
```

## Review Questions

- where is flow control applied first?
- what happens when downstream slows dramatically?
- does buffered work have a hard cap?

## Principal Heuristics

- Bound memory and queue growth before celebrating throughput.
- Prefer explicit shedding or throttling over accidental buffering.
- If you cannot explain where pressure is absorbed, you do not control the stream.
