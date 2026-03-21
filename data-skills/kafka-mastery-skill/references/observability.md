# Observability (Kafka)

## Rules

- Watch lag, throughput, rebalance churn, ISR health, and produce/consume errors.
- Separate broker health from consumer health in dashboards.
- Alert on leading indicators before customer-facing backlog accumulates.
- Trace critical events through producer, topic, and consumer boundaries.

## Observability Heuristics

### Observability should separate semantic distress from infrastructure distress

Good dashboards let teams distinguish broker trouble, producer trouble, consumer lag, replay pressure, and one-topic or one-group pathology quickly.

### Lag needs context to be useful

Lag matters differently depending on topic value, retention posture, consumer semantics, and whether downstream correctness or user-facing freshness is at risk.

### Cluster-wide averages can hide the real incident

One topic, partition, or consumer group may be on fire while overall throughput and cluster medians still look calm.

## Common Failure Modes

### Metric-rich, diagnosis-poor

The platform exposes many numbers, but on-call still cannot tell whether the problem started in producers, brokers, topics, or consumer groups.

### Lag alarm without business meaning

The system pages on lag without explaining whether users, SLAs, or correctness are actually in danger.

### Rebalance pain hidden in generic health views

Consumer churn and instability appear as vague throughput symptoms instead of clearly attributable rebalance behavior.

## Principal Review Lens

- Which metric gives earliest warning of a bad day?
- Can we isolate one broken consumer versus a broker-wide issue fast?
- Are dashboards actionable for on-call, not just impressive?
- Which missing signal would most reduce diagnosis time during a real incident?
