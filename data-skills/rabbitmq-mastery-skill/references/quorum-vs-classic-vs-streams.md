# Quorum vs Classic vs Streams

## Rules

- Queue type choice should follow durability, ordering, and throughput needs.
- Classic queues are not the universal default forever.
- Streams solve different problems than quorum queues.
- Migrations between types need explicit planning.

## Type-Choice Heuristics

### Queue type is a semantics choice first

The right type depends on what the system needs around recovery behavior, ordering expectations, replay posture, storage pattern, and operational complexity.

### Streams are not just “bigger queues”

Streams usually imply different consumption, replay, retention, and operational assumptions than classic or quorum queues.

### Migration cost should shape early decisions

Because moving queue types later can affect publishers, consumers, and operational tooling, the initial choice should be reviewed as an architecture decision rather than a default checkbox.

## Common Failure Modes

### Classic by habit

Teams stay with familiar queue types even when durability or recovery expectations have already outgrown them.

### Streams by hype

The platform adopts streams for novelty or scale rhetoric without a clear need for stream-specific semantics.

### Migration underestimation

The team treats queue-type migration like an infra swap and ignores the client, replay, and operational behavior it can disrupt.

## Principal Review Lens

- Which queue type matches the real failure model?
- Are we using classic queues out of habit?
- What operational cost does the chosen type impose?
- Which current queue-type decision is most likely to be regretted at higher scale or stricter durability needs?
