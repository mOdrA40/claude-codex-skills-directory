# Replication, Sharding, and Reliability

## Rules

- Replication and sharding should follow workload, scale, and operational maturity.
- Distributed tables simplify access but can hide expensive cross-node behavior.
- Recovery, consistency expectations, and replica lag must be understood explicitly.
- High availability should not be assumed from topology diagrams alone.

## Design Guidance

- Choose shard keys based on ingestion and query locality realities.
- Test failure behavior of replicas, distributed queries, and coordination dependencies.
- Keep operator workflows for node loss, rebalance, and replica repair documented.
- Plan around coordination systems and metadata dependencies.

## Principal Review Lens

- Which topology assumption would fail first during node loss?
- Are we sharding for current need or speculative scale theater?
- What cross-shard query pattern is silently expensive?
- Can the team restore healthy replicas without heroics?
