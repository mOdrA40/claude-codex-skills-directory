# Storage, Retention, and Compaction (Kafka)

## Rules

- Retention and compaction policies are part of the product contract.
- Long retention increases storage and recovery burden.
- Compaction semantics must be understood by consumers.
- Tombstones and delete behavior should not surprise downstream teams.

## Principal Review Lens

- Why is this retention period justified?
- What breaks if compaction removes expected history?
- Is storage growth predictable under worst-case traffic?
