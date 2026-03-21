# Workflow Architecture and Event Boundaries (GitHub Actions)

## Rules

- Workflow design should reflect trust boundaries, release stages, and ownership.
- Event triggers are part of your security and reliability model.
- Separate CI, release, maintenance, and privileged automation concerns where possible.
- Avoid workflows that do too many unrelated jobs under one trigger path.

## Design Guidance

- Use clear entry points for pull request validation, mainline integration, and release automation.
- Keep manual, scheduled, and event-driven automation easy to reason about.
- Make environment promotion and approval boundaries explicit.
- Reduce coupling between workflows unless operational leverage justifies it.

## Principal Review Lens

- Which trigger is most likely to fire with unsafe context?
- What workflow boundary is too broad for safe ownership?
- Can reviewers explain what happens on each critical event?
- What architecture simplification would most reduce automation risk?
