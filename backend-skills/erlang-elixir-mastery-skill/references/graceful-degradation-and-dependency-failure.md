# Graceful Degradation and Dependency Failure on the BEAM

## Purpose

Highly available systems are not the ones where dependencies never fail. They are the ones that fail in controlled ways when dependencies slow down, error, or disappear.

## Core Principle

Not every dependency deserves equal treatment.

Classify dependencies as:

- critical for correctness
- critical for freshness but not correctness
- optional enrichment
- asynchronous side-effect only

That classification should determine fallback and degradation behavior.

## Design Rules

- define timeout behavior per dependency class
- isolate optional enrichment from critical request success paths
- avoid broad rescue blocks that hide outage symptoms
- surface degraded mode explicitly in telemetry and logs
- prefer bounded queues over uncontrolled backlog for failing downstreams

## Bad vs Good

```text
❌ BAD
When a recommendation service is slow, the checkout request waits anyway and eventually times out.

✅ GOOD
Checkout completes without recommendations, and the degradation is visible in metrics and logs.
```

## Degradation Options

Choose intentionally:

- serve stale cached data
- skip optional enrichment
- return partial response with explicit semantics
- enqueue side effects for later retry
- fail fast when correctness would otherwise be violated

## Operational Questions

- which dependency failures should fail the request immediately?
- which ones should degrade functionality instead?
- how do operators know the system is degraded but still serving?
- what backlog is acceptable before dropping or shedding work?
- can one failing dependency poison unrelated supervision domains?

## Principal Heuristics

- Do not let optional dependencies define availability of critical workflows.
- Degraded mode should be designed before the incident, not invented during it.
- If degradation hides business-critical correctness issues, it is not graceful.
