# Schema Evolution (Kafka)

## Rules

- Event contracts should evolve deliberately and compatibly.
- Producers and consumers may upgrade at different times.
- Avoid field semantics that change invisibly while schema stays technically valid.
- Versioning must support replay, not only forward traffic.

## Principal Review Lens

- Can old consumers survive this event change?
- What replay path breaks when schema meaning drifts?
- Is compatibility policy explicit and enforced?
