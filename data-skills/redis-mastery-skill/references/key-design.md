# Key Design (Redis)

## Rules

- Key naming should encode domain, tenant, and version where useful.
- Avoid unbounded cardinality and giant values.
- Hot keys need deliberate mitigation.
- Key design affects debugging, memory, and access safety.

## Design Heuristics

### Key shape is part of operability

A good key scheme helps teams understand ownership, tenant scope, invalidation strategy, versioning, and failure blast radius during incidents.

### Encode enough meaning to govern growth

Key design should make it possible to answer:

- who owns this key family
- how big it may grow
- whether it is safe to evict or rebuild
- which tenant or feature is responsible for spikes

### Versioning beats silent semantic drift

When a cached shape or access pattern changes materially, explicit versioning is often safer than letting old and new semantics collide invisibly.

## Common Failure Modes

### Opaque keyspace

The system works until operators need to debug growth, invalidation, or hot-key incidents and cannot tell what a prefix or family really means.

### Shared prefix ambiguity

Different features or services overload one key family until ownership and invalidation become unclear.

### Semantic drift without versioning

The same key name starts carrying new meaning over time, making correctness and rollback harder than expected.

## Principal Review Lens

- Which prefix or keyspace can grow without bound?
- Can operators identify ownership from the key shape?
- Which keys are likely to become hot or huge?
- What key family would be hardest to reason about during an incident today?
