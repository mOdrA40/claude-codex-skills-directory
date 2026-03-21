# Key Design (Redis)

## Rules

- Key naming should encode domain, tenant, and version where useful.
- Avoid unbounded cardinality and giant values.
- Hot keys need deliberate mitigation.
- Key design affects debugging, memory, and access safety.

## Principal Review Lens

- Which prefix or keyspace can grow without bound?
- Can operators identify ownership from the key shape?
- Which keys are likely to become hot or huge?
