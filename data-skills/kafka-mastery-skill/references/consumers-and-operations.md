# Consumers and Operations (Kafka)

## Consumer Defaults

- Tune concurrency to partitions and downstream limits.
- Separate deserialization, validation, business logic, and side effects.
- Commit offsets with intent, not habit.
- Observe lag, rebalance churn, and poison message behavior.

## Operations Defaults

- Monitor broker health, ISR, under-replicated partitions, disk usage, and latency.
- Validate disaster recovery and replay procedures.
- Know the cost of long retention on storage and recovery time.

## Principal Review Lens

- What happens during consumer rebalance in the hottest workload?
- How are poison messages isolated?
- Which lag threshold becomes user-visible pain?
