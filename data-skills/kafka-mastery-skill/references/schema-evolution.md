# Schema Evolution (Kafka)

## Rules

- Event contracts should evolve deliberately and compatibly.
- Producers and consumers may upgrade at different times.
- Avoid field semantics that change invisibly while schema stays technically valid.
- Versioning must support replay, not only forward traffic.

## Evolution Heuristics

### Compatibility must survive time, not just deploy order

Kafka events may be replayed long after producer and consumer versions diverge. Good evolution policy therefore has to survive historical reprocessing, not only near-term rolling upgrades.

### Meaning drift is more dangerous than shape drift

Technically compatible schemas can still be operationally unsafe if field meaning, optionality, or default interpretation changes invisibly across teams.

### Consumer diversity should shape contract discipline

The more teams, runtimes, and replay use cases depend on an event, the stronger the governance needed around schema changes and deprecation windows.

## Common Failure Modes

### Technically valid, semantically broken

The schema passes compatibility checks, but downstream business logic interprets the event differently after the change.

### Replay surprise

Forward traffic works, but old retained events or DLQ replay exposes that the contract was not designed for historical compatibility.

### Hidden consumer population

Teams evolve events assuming they know all consumers, while ad hoc jobs, side systems, or old services still rely on older behavior.

## Principal Review Lens

- Can old consumers survive this event change?
- What replay path breaks when schema meaning drifts?
- Is compatibility policy explicit and enforced?
- Which event contract is currently most likely to be "valid" but still misleading after evolution?
