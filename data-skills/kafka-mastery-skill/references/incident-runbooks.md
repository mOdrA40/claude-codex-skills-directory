# Incident Runbooks (Kafka)

## Rules

- Cover lag storms, broker loss, rebalance thrash, disk pressure, and poison events.
- Stabilize user-visible backlog and critical consumers first.
- Include safe remediation and explicit anti-actions.
- Recovery must be tied to measurable lag and throughput signals.

## Principal Review Lens

- Can on-call isolate the failing topic or consumer quickly?
- Which action risks data loss or wider replay pain?
- What proves recovery versus temporary backlog movement?
