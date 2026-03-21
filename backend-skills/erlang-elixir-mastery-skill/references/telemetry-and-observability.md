# Telemetry and Observability on the BEAM

## Goal

Observability in Erlang and Elixir should help operators answer process, mailbox, dependency, and release questions quickly. Logging alone is insufficient.

## Required Signals

At minimum, design for:

- request rate and latency
- mailbox length for critical processes
- process counts by subsystem
- dependency timeout and failure rates
- queue depth for async work
- scheduler utilization signals
- release version and node identity

## Logging Guidance

Use structured logs with stable fields:

- request_id
- trace_id
- tenant_id when relevant
- node
- module or subsystem
- error class
- dependency name

Never rely on free-form log text as the primary operational contract.

## Telemetry Events

Emit events around:

- request start/stop/exception
- outbound dependency calls
- job processing start/stop/failure
- queue latency and backlog
- supervision restart bursts

## Bad vs Good: Observability Gaps

```text
❌ BAD
The team sees 500s but cannot tell whether failures come from mailboxes, DB pool exhaustion, or external timeouts.

✅ GOOD
Telemetry separates request failure, dependency failure, queue delay, and supervision churn.
```

## Incident Questions

- Which process family is failing most?
- Is one node unhealthy or the whole cluster?
- Are restarts masking overload instead of fixing it?
- Are mailboxes growing faster than workers can drain?

## Review Checklist

- Critical process families are measurable.
- Dependency calls emit latency and error signals.
- Node identity is present in logs/metrics.
- Release and rollout diagnosis is possible from telemetry.
