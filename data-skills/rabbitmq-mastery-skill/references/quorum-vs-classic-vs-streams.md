# Quorum vs Classic vs Streams

## Rules

- Queue type choice should follow durability, ordering, and throughput needs.
- Classic queues are not the universal default forever.
- Streams solve different problems than quorum queues.
- Migrations between types need explicit planning.

## Principal Review Lens

- Which queue type matches the real failure model?
- Are we using classic queues out of habit?
- What operational cost does the chosen type impose?
