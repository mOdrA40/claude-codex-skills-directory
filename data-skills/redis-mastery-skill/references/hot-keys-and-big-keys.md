# Hot Keys and Big Keys (Redis)

## Rules

- Hot keys create latency cliffs and uneven scaling.
- Big keys create memory, CPU, and operational pain.
- Detect them before incidents via telemetry and sampling.
- Mitigation usually needs design changes, not only bigger boxes.

## Principal Review Lens

- Which key becomes hottest during peak business events?
- What key shape creates worst-case serialization cost?
- Can one tenant dominate a shared keyspace?
