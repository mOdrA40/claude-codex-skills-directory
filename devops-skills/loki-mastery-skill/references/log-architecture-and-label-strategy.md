# Log Architecture and Label Strategy (Loki)

## Rules

- Labels should be bounded, stable, and useful for narrowing search space.
- Unbounded identifiers belong in log content, not label space.
- Log architecture should reflect tenant, environment, service, and operational purpose.
- Label strategy is the primary cost and scale control lever in Loki.

## Design Guidance

- Standardize core labels across teams.
- Keep application and platform labels distinct where it improves query clarity.
- Review label proposals the way you review metrics dimensions.
- Make high-cardinality failure modes visible before production rollout.

## Principal Review Lens

- Which label is most likely to explode in six months?
- Are we modeling operational questions or just indexing randomness?
- What standardization gap causes most logging confusion today?
- Which label should be removed to cut cost without hurting investigations?
