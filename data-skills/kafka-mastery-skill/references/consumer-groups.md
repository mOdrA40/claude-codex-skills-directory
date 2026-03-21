# Consumer Groups (Kafka)

## Rules

- Group design should reflect ownership of work, not arbitrary team boundaries.
- Rebalance behavior must be understood on hot workloads.
- Concurrency should match partitions and downstream capacity.
- Offset commit strategy should be deliberate and observable.

## Principal Review Lens

- What happens to latency and duplicates during rebalance?
- Are partitions or downstream bottlenecks limiting throughput?
- Is one group trying to do too many different jobs?
