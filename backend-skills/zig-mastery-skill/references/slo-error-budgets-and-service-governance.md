# SLO, Error Budgets, and Service Governance for Zig Services

## Purpose

Fast services are not automatically reliable services. At scale, principal-level engineering requires clear service objectives, explicit tradeoffs, and rules for when feature velocity must yield to reliability work.

## Core Principle

If a Zig service cannot answer what good reliability looks like, it cannot make principled overload, retry, or rollout decisions.

Define at least:

- critical user journeys
- latency objectives
- availability objectives
- acceptable degradation modes
- ownership and escalation path

## SLI Candidates

Pick indicators tied to real user outcomes, not vanity numbers.

Examples:

- successful request rate for critical routes
- p95 and p99 latency for read and write paths separately
- dependency success rate for mandatory downstreams
- queue age for workflows with asynchronous completion guarantees
- freshness lag for derived or replicated data

## Error Budget Discipline

An error budget is not a dashboard ornament. It is a decision tool.

When the service is burning budget too quickly:

- slow or pause risky releases
- prioritize reliability defects over feature work
- reduce optional load or expensive features
- tighten change review for hot paths
- add mitigation for the dominant burn source

## Governance Questions

- who owns the SLO definition?
- which routes or workflows actually matter to the business?
- what degradation is acceptable before the SLO is considered violated?
- what release gates should trigger when budget burn spikes?
- which dependencies are inside versus outside the service promise?

## Bad vs Good

```text
❌ BAD
The service has one generic uptime goal, but no route-level latency objectives, no budget policy, and no clear response when incidents recur.

✅ GOOD
Critical workflows have explicit SLOs, budget burn is visible, and release/change policy adjusts when the service is unstable.
```

## Principal Heuristics

- Prefer a few meaningful SLOs over many noisy ones.
- Tie observability and runbooks to SLOs, not the other way around.
- If the team never changes behavior when the error budget is consumed, the governance model is fake.
