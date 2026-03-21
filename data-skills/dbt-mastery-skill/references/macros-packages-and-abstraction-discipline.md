# Macros, Packages, and Abstraction Discipline

## Rules

- Macros should reduce repetition without obscuring SQL intent.
- Shared packages are platform dependencies and require governance.
- Abstraction that hides semantics creates long-term analytics debt.
- Reuse should preserve reviewability and debugging clarity.

## Practical Guidance

- Keep macro interfaces narrow and meaningful.
- Review package upgrades for wide model impact.
- Avoid macro-heavy designs that make compiled SQL the only source of truth.
- Document where abstraction is required versus merely clever.

## Principal Review Lens

- Which macro hides the most business logic today?
- Are we standardizing wisely or centralizing confusion?
- What shared package has the highest blast radius?
- Which abstraction should be simplified first?
