# Consumer Groups (Kafka)

## Rules

- Group design should reflect ownership of work, not arbitrary team boundaries.
- Rebalance behavior must be understood on hot workloads.
- Concurrency should match partitions and downstream capacity.
- Offset commit strategy should be deliberate and observable.

## Group Heuristics

### Consumer groups are workflow boundaries

A group defines not only concurrency but also replay scope, ownership of lag, rebalance behavior, and how work is shared or isolated operationally.

### Rebalance cost should be treated as real user pain

On hot or latency-sensitive workloads, rebalance events are not minor implementation details. They directly affect throughput, duplicate risk, and backlog growth.

### Offset strategy should match business effect timing

The right commit posture depends on when side effects become safe, what duplicates are tolerable, and how much replay ambiguity the workflow can absorb.

## Common Failure Modes

### One group, too many jobs

A single group ends up serving multiple different operational intents, making lag, replay, and failure handling much harder to reason about.

### Rebalance normalized as background noise

The platform accepts recurring rebalance pain without redesigning membership churn, deployment behavior, or workload partitioning.

### Commit semantics by habit

Offsets are committed according to framework defaults or copied patterns rather than real business-side safety needs.

## Principal Review Lens

- What happens to latency and duplicates during rebalance?
- Are partitions or downstream bottlenecks limiting throughput?
- Is one group trying to do too many different jobs?
- Which consumer-group assumption is most likely to fail under deploy churn or traffic spikes?
