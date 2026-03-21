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

## Consumer Heuristics

### Consumers should separate correctness stages explicitly

The most reliable Kafka consumers make it clear where deserialization ends, validation begins, business logic executes, and external side effects happen. That separation makes replay, retry, and incident diagnosis safer.

### Offset handling should follow business semantics

Commit timing is not just a client detail. It determines what duplicates, replay scope, and partial failure behavior the workflow can tolerate.

### Hot consumers need operational identity

For important workloads, operators should know which group, partition set, and downstream dependency are responsible when lag or replay risk begins to grow.

## Common Failure Modes

### Consumer loop opacity

The service processes messages successfully enough in normal traffic, but teams cannot explain exactly where poison events, duplicate side effects, or slow downstream interactions should be isolated.

### Offset semantics by default

Client library defaults or copied patterns decide commit behavior more than actual business-side recovery needs.

### Operations and consumer logic disconnected

Broker health is monitored separately from consumer correctness, so incidents move slowly from infrastructure symptoms to business understanding.

## Principal Review Lens

- What happens during consumer rebalance in the hottest workload?
- How are poison messages isolated?
- Which lag threshold becomes user-visible pain?
- Which consumer-stage boundary is currently least explicit and most dangerous under replay or failure?
