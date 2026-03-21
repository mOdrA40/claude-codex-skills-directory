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

## Consumer Heuristics

### Ack semantics should match side-effect reality

The right ack point depends on what is actually durable and what duplicate behavior the workflow can tolerate.

### Replay is an operational event

Replaying messages should be treated like a load and correctness event, not merely a convenience feature.

### Separate consumer classes by risk

Not every workload should share one ack, concurrency, and replay policy. Some consumers prioritize low latency, others durability, others controlled side effects.

## Additional Failure Modes

### Replay optimism

The team assumes replay is safe because the messaging layer supports it, but downstream services, rate limits, or side effects were never designed for the replay load.

### Semantic ambiguity

Operators can see lag and redelivery, but they still cannot tell what business risk those signals imply.

## Principal Review Lens

- What duplicate behavior is acceptable for this workflow?
- Which side effect breaks first under replay?
- Can operators reason about lag and redelivery quickly?
- Are consumer semantics clearly owned by the service team?
- Which consumer should be redesigned before a replay event turns into an incident?
