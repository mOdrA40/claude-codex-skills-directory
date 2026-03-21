# Topic Design and Delivery (Kafka)

## Topic Rules

- Topics model durable business streams, not arbitrary team boundaries.
- Partition keys should preserve needed ordering while distributing load.
- Schema evolution must be backward/forward compatible where required.
- Retention and compaction should match replay and recovery needs.

## Delivery Rules

- Consumers must be idempotent.
- Avoid claiming exactly-once where system boundaries still duplicate work.
- Make DLQ/retry policy explicit.

## Principal Review Lens

- What is the event contract and how does it evolve?
- Which keys risk hot partitions?
- What duplicate or out-of-order behavior must consumers tolerate?
