# Observability (Kafka)

## Rules

- Watch lag, throughput, rebalance churn, ISR health, and produce/consume errors.
- Separate broker health from consumer health in dashboards.
- Alert on leading indicators before customer-facing backlog accumulates.
- Trace critical events through producer, topic, and consumer boundaries.

## Principal Review Lens

- Which metric gives earliest warning of a bad day?
- Can we isolate one broken consumer versus a broker-wide issue fast?
- Are dashboards actionable for on-call, not just impressive?
