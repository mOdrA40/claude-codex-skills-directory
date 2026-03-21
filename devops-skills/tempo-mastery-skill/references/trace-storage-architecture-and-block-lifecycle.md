# Trace Storage Architecture and Block Lifecycle (Tempo)

## Rules

- Storage design should reflect query habits, trace volume, and retention goals.
- Block lifecycle behavior affects cost, availability, and operator understanding.
- Object storage dependencies are part of Tempo's reliability model.
- Simplicity in storage architecture often beats premature optimization.

## Design Guidance

- Understand ingest-to-block flow and compaction implications.
- Match architecture to expected search patterns and retention windows.
- Keep storage decisions aligned with incident needs, not only cost goals.
- Document where trace truth lives during each stage of the lifecycle.

## Principal Review Lens

- What storage assumption is least tested today?
- Which lifecycle stage hides the most operational risk?
- Are we optimizing for trace durability, searchability, or both clearly?
- What failure would make traces appear present but unusable?
