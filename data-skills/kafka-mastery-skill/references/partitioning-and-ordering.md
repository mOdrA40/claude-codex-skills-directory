# Partitioning and Ordering (Kafka)

## Rules

- Partition by stable business keys that align with ordering needs.
- Ordering is only guaranteed within a partition.
- Hot partitions are product and key-design failures, not mere tuning issues.
- Repartitioning later is possible but expensive.

## Design Heuristics

### Partition keys encode scale strategy

A partition key is not just a routing detail. It decides where ordering holds, where load concentrates, and how expensive future redistribution becomes.

### Ask what truly needs ordering

Many systems over-demand ordering and pay for it with hot partitions, constrained scale, and awkward recovery behavior.

### Future skew matters more than current balance

The dangerous question is not whether the key distributes well today, but whether product growth, tenant concentration, or time-based events will break it later.

## Common Failure Modes

### Ordering by habit

The team chooses a key that preserves more ordering than the business actually needs, reducing scale headroom unnecessarily.

### Balanced in test, skewed in reality

A partition strategy appears fine in average traffic but collapses around one tenant, entity, or event class under real business peaks.

### Repartition pain underestimated

The original design treats future partition changes as manageable, but migration cost, consumer impact, and ordering semantics make change much harder than assumed.

## Principal Review Lens

- Which entity actually needs ordering guarantees?
- Which key could create partition skew under peak traffic?
- Are we over-constraining ordering and hurting scale?
- What partition key assumption will age worst if adoption or tenant skew grows sharply?
