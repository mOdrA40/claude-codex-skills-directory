# Incident Runbooks (CockroachDB)

## Rules

- Runbooks should cover contention storms, regional impairment, node loss, and latency regressions.
- Start with blast-radius reduction before perfect diagnosis.
- Include safe operator actions and forbidden shortcuts.
- Tie recovery steps to measurable signals.

## Incident Classes

Useful runbooks should distinguish between:

- contention-driven latency and restart storms
- locality or zone-placement mistakes
- node or region impairment
- schema-change and backfill side effects
- operator actions that look safe but worsen placement or load balance

## Operator Heuristics

### Stabilize first

Initial questions should focus on:

- which workload class is failing
- whether the problem is cluster-wide or locality-specific
- whether write concentration, placement, or background work changed recently

### Avoid accidental escalation

Runbooks should explicitly call out actions that may worsen matters, such as forcing broad operational changes without understanding placement, lease, or workload shape.

### Recovery must be measurable

The runbook should name the metrics or symptoms that prove recovery, not just "traffic seems calmer now".

## Common Failure Modes

### Generic database runbooks

Teams use a generic SQL-database incident playbook and miss distributed placement, leaseholder, or retry-behavior causes.

### Temporary calm mistaken for recovery

Retries fall for a moment or one dashboard looks better, but the underlying contention or locality problem remains.

## Principal Review Lens

- Can on-call stabilize retry storms quickly?
- Which action risks making placement or availability worse?
- What metric proves real recovery rather than temporary calm?
- What incident class still lacks an explicit safe-first operator sequence?
