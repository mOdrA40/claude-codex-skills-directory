# Capacity Planning and Load Shedding in Rust Services

## Purpose

Rust gives strong control over memory and concurrency, but high-scale systems still fail if they do not define capacity posture explicitly. The goal is not only to survive benchmarks. The goal is to preserve critical workflows under real overload.

## Core Principle

Capacity planning means understanding what saturates first and what traffic should be protected when it does.

## Questions to Answer

- what saturates first: CPU, allocator churn, lock contention, DB pool, queue age, downstream rate limit?
- which endpoints or jobs are highest value?
- what is normal burst versus dangerous burst?
- what can be dropped, degraded, queued, or rejected?
- how does one tenant affect everyone else?

## Load Shedding Options

Choose intentionally:

- reject low-priority work early
- reduce fan-out or optional enrichment
- enforce tighter concurrency caps
- rate-limit by tenant or traffic class
- stop accepting expensive background work temporarily

## Review Questions

- does the service fail fast or slowly drown in buffers and queues?
- are retries amplifying overload across layers?
- is backpressure visible in logs and metrics before saturation becomes outage?
- which work must be preserved for business correctness?
- which dependency limit defines the real ceiling?

## Principal Heuristics

- Protect critical workflows before preserving headline throughput.
- Capacity claims without dependency limits are incomplete.
- If overload policy begins only after memory or queue collapse, it is too late.
