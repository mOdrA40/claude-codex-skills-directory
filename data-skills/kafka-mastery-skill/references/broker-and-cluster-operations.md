# Broker and Cluster Operations (Kafka)

## Rules

- Monitor ISR, under-replicated partitions, disk, network, and controller health.
- Operational changes should be staged with partition movement cost in mind.
- Cluster upgrades and reassignments must be boring and reversible.
- Capacity headroom should include failure scenarios, not only normal load.

## Operational Heuristics

### Broker health is about recovery posture, not just uptime

The real question is whether the cluster can absorb maintenance, broker failure, partition movement, and replay pressure without surprising application teams.

### Reassignments are workload events

Partition movement and maintenance work should be treated like controlled production load because they affect network, disk, ISR stability, and client experience.

### Operational simplicity is an architecture asset

The safest Kafka platform is one where controller behavior, broker roles, and maintenance steps remain understandable under stress.

## Common Failure Modes

### Healthy-looking brokers, weak maintenance posture

The cluster appears fine in steady state, but routine operations like rebalance, expansion, or upgrade create too much instability.

### Partition-movement optimism

Teams assume reassignments are safe because the tooling exists, without respecting the real bandwidth and recovery tax.

### Hidden single-broker dependency

One broker or storage path quietly becomes far more critical than intended for certain hot partitions or workloads.

## Principal Review Lens

- Which broker failure causes largest blast radius?
- Are reassignments and maintenance safe under current utilization?
- What hidden single point of failure still exists operationally?
- Which operational activity is most likely to turn a healthy cluster into a self-inflicted incident?
