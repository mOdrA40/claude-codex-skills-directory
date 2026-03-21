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

## Design Heuristics

### Topics are product boundaries

A topic should represent a durable stream with a clear contract, ownership model, lifecycle, and replay value—not just a convenient place to publish data.

### Delivery posture should be designed with topic purpose

The right delivery, retry, and retention strategy depends on what the topic means, how consumers evolve, and how harmful duplicates or reordering are for that specific stream.

### Topic count and topic quality both matter

Too few topics can blur semantics and ownership. Too many weakly governed topics create sprawl, confusion, and platform debt.

## Common Failure Modes

### Topic by team chart

Topics are created around org structure rather than durable business semantics, producing awkward ownership and replay stories later.

### Delivery semantics overgeneralized

One retry or DLQ pattern is copied broadly even though different topics imply different risk and consumer expectations.

### Contract drift hidden by continued publishing

The topic still flows, but ownership, meaning, and consumer expectations are already diverging dangerously.

## Principal Review Lens

- What is the event contract and how does it evolve?
- Which keys risk hot partitions?
- What duplicate or out-of-order behavior must consumers tolerate?
- Which topic today is publishing successfully while semantically drifting the most?
