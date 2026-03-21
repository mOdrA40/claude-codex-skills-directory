# Subject Design and Routing Semantics (NATS)

## Rules

- Subject hierarchy should reflect bounded domains and stable routing intent.
- Wildcards are powerful but can create unintended coupling if overused.
- Naming should make ownership and message class obvious.
- Subject design is part of platform governance, not local taste.

## Design Guidance

- Keep subject tokens meaningful and bounded.
- Avoid embedding unbounded identifiers in subject space without a good reason.
- Separate command-like patterns, events, and operational control subjects clearly.
- Standardize naming conventions early in shared platforms.

## Routing Heuristics

### Subject design is an API surface

Subjects communicate ownership, routing semantics, wildcard safety, and what kinds of consumers or operators may reasonably subscribe.

### Wildcards should not replace architecture

Wildcard power is useful, but when teams lean on it too heavily they often avoid making clearer domain boundaries and routing commitments.

### Subject sprawl becomes governance debt quickly

As team count grows, weak conventions make discoverability, policy control, auditing, and incident response harder than they need to be.

## Common Failure Modes

### Naming by local convenience

Subjects work for the first team that created them but become confusing or unsafe once shared across multiple services or operators.

### Wildcard coupling

Consumers depend on broad patterns that make future routing changes, ownership enforcement, or access control much harder.

### Semantic drift in subject space

The same naming family starts carrying multiple meanings over time, making routing intent and incident diagnosis harder.

## Principal Review Lens

- Which subject pattern will create governance pain as teams grow?
- Are wildcards being used for flexibility or to avoid real design?
- Can operators infer traffic ownership from subject names?
- What subject convention would most reduce future chaos?
- Which subject family looks simple now but will become the hardest to govern later?
