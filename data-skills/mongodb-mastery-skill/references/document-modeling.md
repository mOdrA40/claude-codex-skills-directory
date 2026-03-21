# Document Modeling (MongoDB)

## Rules

- Model for dominant read/write paths first.
- Embedding is good when lifecycle and access are shared.
- Referencing is good when cardinality, independence, or growth demands it.
- Unbounded arrays and runaway document growth are production traps.

## Principal Review Lens

- What query shape drove this model?
- How will this document evolve over 12 months?
- Which write path becomes expensive as the document grows?
