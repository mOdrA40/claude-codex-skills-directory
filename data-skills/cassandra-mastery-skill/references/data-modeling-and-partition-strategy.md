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

## Principal Review Lens

- Which partition key is most likely to create a hotspot?
- Are we modeling for query efficiency or carrying over SQL instincts?
- What data growth pattern makes today's partitioning unsafe?
- Which table design most deserves redesign before production scale?
