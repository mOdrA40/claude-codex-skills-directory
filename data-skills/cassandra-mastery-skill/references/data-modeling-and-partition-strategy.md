# Data Modeling and Partition Strategy (Cassandra)

## Rules

- Model by query and access path, not by relational normalization habits.
- Partition keys determine scale behavior, hotspot risk, and operational pain.
- Clustering columns should support predictable sort and slice patterns.
- Avoid designs that create oversized partitions or fan-out query pain.

## Design Guidance

- Identify the hottest read/write paths before schema design.
- Validate expected partition cardinality and growth over time.
- Duplicate data intentionally when it simplifies query paths.
- Make tenant and time dimensions explicit where they shape load.

## Modeling Heuristics

### Partition keys are workload contracts

In Cassandra, a partition key defines not only lookup shape but also hotspot risk, repair behavior, storage growth, and operator pain later.

### Design for the hottest path, not average elegance

A model that looks tidy on paper can still be unsafe if one tenant, one time bucket, or one access pattern dominates real production traffic.

### Denormalization should reduce operational ambiguity

Copying data is justified when it makes query paths stable and cheap, not when it creates unclear ownership or hidden consistency risk.

## Common Failure Modes

### SQL instincts in wide-column clothing

The schema imitates relational instincts and pushes complexity into fan-out reads, oversized partitions, or awkward access paths.

### Average cardinality comfort

The model looks safe by average partition size while worst-case tenants or event bursts create the real hotspot danger.

### Time-bucket denial

Time dimensions help access patterns initially, then become the main source of uneven pressure and partition pain as usage grows.

## Principal Review Lens

- Which partition key is most likely to create a hotspot?
- Are we modeling for query efficiency or carrying over SQL instincts?
- What data growth pattern makes today's partitioning unsafe?
- Which table design most deserves redesign before production scale?
- Which partition assumption is least likely to survive asymmetric tenant growth?
