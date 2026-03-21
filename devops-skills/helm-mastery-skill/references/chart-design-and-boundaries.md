# Chart Design and Boundaries (Helm)

## Rules

- A chart should package one coherent deployment concern, not every cluster concern at once.
- Shared library charts are useful when they simplify consistency without hiding behavior.
- Chart boundaries should match ownership, release cadence, and rollback needs.
- Avoid charts that become unreviewable policy engines.

## Design Guidance

- Separate infrastructure add-ons from application release charts.
- Prefer explicit templates and named helpers over sprawling abstraction layers.
- Make resource names, labels, and selectors predictable.
- Keep dependencies visible and justified.

## Principal Review Lens

- Does this chart reduce duplication or merely relocate complexity?
- What chart boundary is fake because ownership is already separate?
- Can operators predict rendered resources without executing mental gymnastics?
- Which reuse decision today becomes tomorrow's upgrade trap?
