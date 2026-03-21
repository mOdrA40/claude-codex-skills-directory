# Document Modeling (MongoDB)

## Rules

- Model for dominant read/write paths first.
- Embedding is good when lifecycle and access are shared.
- Referencing is good when cardinality, independence, or growth demands it.
- Unbounded arrays and runaway document growth are production traps.

## Modeling Heuristics

### Embed when the lifecycle is truly shared

Embedding works best when reads, writes, and ownership really move together. If subdocuments evolve independently, embedding often becomes expensive or operationally awkward.

### Reference when scale and independence matter

Referencing is usually safer when:

- cardinality grows without clear upper bounds
- one sub-entity is updated much more frequently than its parent
- retention or security policy differs across entities

### Model growth over time, not just today

A document shape that feels elegant at launch can become painful once arrays expand, hot fields update frequently, or access patterns diverge.

## Common Failure Modes

### Model by aesthetic preference

Teams choose embedding or referencing because it feels cleaner, not because the main read/write path and growth pattern justify it.

### Hidden write amplification

One logical business update rewrites or contends on far more document surface than intended.

### Array optimism

An array starts small and convenient, then slowly becomes the main source of document bloat, write pain, and awkward query behavior.

## Principal Review Lens

- What query shape drove this model?
- How will this document evolve over 12 months?
- Which write path becomes expensive as the document grows?
- What part of this model looks elegant now but becomes a trap under scale or product evolution?
