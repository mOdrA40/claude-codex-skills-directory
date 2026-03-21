# Reusable Workflows and Action Governance

## Rules

- Reuse should reduce duplication without hiding critical delivery behavior.
- Shared actions and reusable workflows are platform dependencies and need versioning discipline.
- Pinning and provenance matter for third-party and internal actions alike.
- Governance should constrain unsafe or ownerless automation reuse.

## Practical Guidance

- Define platform-approved workflows and action sets.
- Keep interfaces narrow and semantically meaningful.
- Review upgrade blast radius for widely used actions.
- Track ownership of shared automation artifacts.

## Principal Review Lens

- Which shared workflow has the highest blast radius today?
- Are teams reusing trusted building blocks or copy-pasting drift?
- What hidden behavior is buried in reusable workflow abstraction?
- Which governance rule would most improve CI/CD safety?
