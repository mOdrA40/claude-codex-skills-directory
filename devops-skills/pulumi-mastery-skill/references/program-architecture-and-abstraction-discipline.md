# Program Architecture and Abstraction Discipline

## Rules

- Programmatic IaC should improve clarity, not create opaque logic towers.
- Abstractions must reduce repetition while preserving operational meaning.
- General-purpose language features should be used with discipline.
- Infrastructure behavior should remain inferable without executing mental gymnastics.

## Common Failure Modes

- Overuse of inheritance, factories, or condition-heavy code that hides resource intent.
- Business logic leaking into infrastructure decisions without clear ownership.
- Clever helpers that generate resources nobody can review safely.
- Testing focus on code style while missing deployment semantics.

## Principal Review Lens

- Can an infra reviewer understand what this program creates?
- Which abstraction hides the most operational risk?
- Are we writing maintainable platform code or software theater?
- What simplification would most improve review confidence?
