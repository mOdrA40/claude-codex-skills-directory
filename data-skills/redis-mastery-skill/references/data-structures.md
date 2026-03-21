# Data Structures (Redis)

## Rules

- Use strings, hashes, sets, sorted sets, streams, and bitmaps for clear reasons.
- Data structure choice affects memory, query shape, and ops behavior.
- Avoid accidental complexity by mixing too many primitives per workflow.
- Model lifecycle and eviction with the structure in mind.

## Principal Review Lens

- Is this structure optimizing the real access path?
- What operation becomes expensive at scale?
- Are we using Redis like a database without admitting it?
