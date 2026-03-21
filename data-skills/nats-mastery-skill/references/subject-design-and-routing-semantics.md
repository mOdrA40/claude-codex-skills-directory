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

## Principal Review Lens

- Which subject pattern will create governance pain as teams grow?
- Are wildcards being used for flexibility or to avoid real design?
- Can operators infer traffic ownership from subject names?
- What subject convention would most reduce future chaos?
