# Lineage, Ownership, and Data Product Governance

## Rules

- Lineage should support operational accountability, not just pretty graphs.
- Ownership of source, model, and consumer-facing datasets must be explicit.
- Data product governance should reduce ambiguity around change and support.
- Platform scale requires standard ownership and deprecation behavior.

## Practical Guidance

- Track which teams own upstream sources and downstream contracts.
- Use lineage to expose blast radius of model changes.
- Make deprecation, migration, and breaking change workflows explicit.
- Keep critical downstream consumers visible to maintainers.

## Principal Review Lens

- Which model has the highest blast radius and weakest ownership?
- Are lineage artifacts actually helping decisions?
- What contract is missing between producer and consumer teams?
- Which governance change most improves trust and maintainability?
