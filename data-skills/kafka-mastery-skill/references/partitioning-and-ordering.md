# Partitioning and Ordering (Kafka)

## Rules

- Partition by stable business keys that align with ordering needs.
- Ordering is only guaranteed within a partition.
- Hot partitions are product and key-design failures, not mere tuning issues.
- Repartitioning later is possible but expensive.

## Principal Review Lens

- Which entity actually needs ordering guarantees?
- Which key could create partition skew under peak traffic?
- Are we over-constraining ordering and hurting scale?
