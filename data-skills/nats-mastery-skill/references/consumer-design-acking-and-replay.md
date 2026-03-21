# Consumer Design, Acking, and Replay

## Rules

- Consumer ack behavior should match processing semantics and duplicate tolerance.
- Replay is powerful and dangerous; side effects must be designed accordingly.
- Durable consumers need ownership and operational visibility.
- Redelivery behavior should be expected, not treated as impossible.

## Common Mistakes

- Assuming ack means business side effects are safely committed everywhere.
- Using one consumer design across fundamentally different workloads.
- Ignoring redelivery until duplicates become an incident.
- Letting replay overload downstream dependencies.

## Principal Review Lens

- What duplicate behavior is acceptable for this workflow?
- Which side effect breaks first under replay?
- Can operators reason about lag and redelivery quickly?
- Are consumer semantics clearly owned by the service team?
