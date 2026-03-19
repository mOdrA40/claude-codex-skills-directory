# Incident Runbooks (Go Services)

This guide is for the first 30 minutes of a production incident: contain blast radius, gather evidence, restore service, and only then optimize.

## First Principles

- Stabilize first, diagnose second, refactor last.
- Prefer reducing blast radius over chasing perfect root cause in the middle of the outage.
- Preserve evidence: logs, trace IDs, dashboards, deploy SHA, config version.
- Every mitigation should be reversible.

## Triage Checklist

- What changed recently: deploy, config, migration, dependency outage, traffic spike?
- What is failing: all requests, one endpoint, one tenant, one region, one dependency?
- Is the system overloaded, degraded, or outright broken?
- What is the customer-visible symptom: errors, latency, stale data, duplicate side effects?

## Minimum Evidence to Capture

- Current deploy SHA / version.
- Error rate, p95/p99 latency, saturation indicators.
- At least one representative trace ID or request ID.
- Dependency health: DB pool wait time, queue lag, outbound timeout/error rate.
- Any rollback, feature-flag, or traffic-shift actions taken.

## Common Failure Modes

## 1. Latency spike

Check:

- DB pool exhaustion.
- Retry storms.
- Slow downstream dependency.
- Unbounded goroutines or queues.
- Recent config change on timeouts or limits.

Immediate mitigations:

- Reduce concurrency.
- Shed load on expensive endpoints.
- Disable non-essential fan-out.
- Tighten timeouts to fail fast.

## 2. Error-rate spike

Check:

- Recent deploy or migration.
- Expired credentials or secret rotation issues.
- Schema drift between app and DB.
- Strict validation changes rejecting valid traffic.

Immediate mitigations:

- Roll back if the regression is new and obvious.
- Disable the affected feature path behind a flag if possible.
- Return explicit `503` for degraded dependency paths instead of hanging.

## 3. Duplicate side effects

Check:

- Retries at multiple layers.
- Missing or broken idempotency keys.
- Consumer re-delivery after partial success.
- Outbox publisher replay without deduplication.

Immediate mitigations:

- Pause consumers if duplication is still ongoing.
- Enforce temporary deduplication at the storage boundary.
- Disable retries on non-idempotent paths.

## 4. Memory or goroutine leak

Check:

- Goroutine count trend.
- Heap growth trend and GC pause time.
- Background loops without stop conditions.
- Stuck outbound calls without deadlines.

Immediate mitigations:

- Restart only after evidence is captured.
- Reduce traffic or disable the leaking feature path.
- Turn on pprof or capture profiles if safe.

## Rollback Decision Framework

Roll back when all are true:

- The issue started after a recent change.
- The rollback is low risk and well understood.
- The system can tolerate temporary feature loss better than ongoing outage.

Do not roll back blindly when:

- The issue is caused by data corruption or irreversible schema changes.
- The real problem is external dependency failure.
- Rollback would worsen state divergence.

## Post-Incident Questions

- Which alert should have fired earlier?
- Which signal was noisy or missing?
- Which runbook step was unclear?
- What safe default would have reduced the blast radius?
- What test or chaos drill would have caught this before production?

## Principal Lens

A principal-grade service is not one that never fails. It is one that:

- fails in known ways,
- exposes the right signals,
- supports safe rollback,
- and lets the team restore service without heroics.
