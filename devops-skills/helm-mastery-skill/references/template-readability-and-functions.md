# Template Readability and Functions

## Rules

- Template logic should remain readable under pressure.
- Named templates and helper functions should reduce duplication, not obscure control flow.
- Deep nesting, excessive conditionals, and magic defaults create review hazards.
- Rendering should be deterministic and easy to reason about.

## Common Mistakes

- Turning Helm into a programming language because the team can.
- Hiding critical behavior inside helpers with unclear names.
- Using conditionals to support too many incompatible deployment shapes.
- Making output correctness dependent on undocumented value combinations.

## Principal Review Lens

- Can a reviewer explain what this template renders without running it?
- Which helper hides behavior that matters operationally?
- Are we paying readability to support edge cases nobody needs?
- What simplification would most improve trust in this chart?
