# Capacity Planning and Load Shedding in Bun Services

## Purpose

Bun can serve fast APIs, but high-scale systems fail when they lack deliberate capacity posture. Principal-level backend work requires knowing where the service saturates, how it degrades, and what gets protected first.

## Core Principle

Capacity planning is not predicting one magic QPS number. It is understanding which resources saturate first and how to keep failure controlled when they do.

## Capacity Questions

- what saturates first: CPU, memory, connection pools, queue depth, downstream rate limits?
- what is normal burst versus dangerous burst?
- what is the admission policy when safe concurrency is exceeded?
- which traffic classes are most valuable to preserve?
- what are the per-tenant or per-endpoint hotspots?

## Load Shedding Options

Choose intentionally:

- reject low-priority traffic early
- enforce tighter per-tenant or per-route limits
- degrade optional enrichments
- stop accepting expensive fan-out work
- fail readiness to drain traffic from a bad instance

## Bad vs Good

```text
❌ BAD
Every request is treated equally until Node/Bun workers, DB pools, and downstreams all saturate together.

✅ GOOD
Traffic classes and expensive paths are known, overload is detected early, and low-value work is shed before critical workflows collapse.
```

## Operational Signals

Track at least:

- in-flight requests
- event loop delay or equivalent saturation signal
- DB/Redis pool acquisition latency
- queue depth and oldest work age
- p95/p99 latency by endpoint class
- rate-limited or shed requests

## Principal Heuristics

- Protect critical workflows before overall throughput optics.
- If overload policy starts only after hard saturation, it is too late.
- Capacity planning must include dependency limits, not only app-server benchmarks.
