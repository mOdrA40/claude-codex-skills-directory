# Broker and Cluster Operations (Kafka)

## Rules

- Monitor ISR, under-replicated partitions, disk, network, and controller health.
- Operational changes should be staged with partition movement cost in mind.
- Cluster upgrades and reassignments must be boring and reversible.
- Capacity headroom should include failure scenarios, not only normal load.

## Principal Review Lens

- Which broker failure causes largest blast radius?
- Are reassignments and maintenance safe under current utilization?
- What hidden single point of failure still exists operationally?
