# Data Structures (Redis)

## Rules

- Use strings, hashes, sets, sorted sets, streams, and bitmaps for clear reasons.
- Data structure choice affects memory, query shape, and ops behavior.
- Avoid accidental complexity by mixing too many primitives per workflow.
- Model lifecycle and eviction with the structure in mind.

## Structure Heuristics

### Every structure encodes a workload bet

Choosing hashes, sets, sorted sets, streams, or strings is really choosing mutation cost, lookup shape, memory growth behavior, and debugging complexity.

### Simplicity usually wins under operational stress

The best Redis structure is often the one that keeps correctness and observability easiest to explain, even if another structure looks slightly more elegant in code.

### Lifecycle and eviction should shape the choice early

If a structure is hard to expire, bound, inspect, or rebuild safely, it is often the wrong fit for a production Redis workload.

## Common Failure Modes

### Structure by familiarity

The team chooses the primitive they know best rather than the one that best matches memory, mutation, and recovery needs.

### Mixed-primitive workflow sprawl

One business flow relies on too many Redis primitives, making debugging, failure handling, and ownership much harder.

### Unbounded-structure surprise

The chosen structure works well initially, then grows in ways that make memory and operational behavior much worse than expected.

## Principal Review Lens

- Is this structure optimizing the real access path?
- What operation becomes expensive at scale?
- Are we using Redis like a database without admitting it?
- Which data-structure choice is most likely to create hidden operational pain later?
