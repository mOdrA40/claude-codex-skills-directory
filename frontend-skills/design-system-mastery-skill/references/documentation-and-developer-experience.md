# Documentation and Developer Experience

## Principle

A design system with weak documentation becomes tribal knowledge. Good DX is part of the adoption strategy, not a polish step.

## DX Model

Great design-system DX helps engineers answer quickly:

- which component to use
- when not to use it
- how to compose it safely
- what accessibility guarantees it carries
- how upgrades affect them

If those answers require asking a senior or reading source code, the system is under-documented.

## Must-Have Areas

- usage guidance
- do and do-not examples
- accessibility notes
- migration guidance
- visual and behavioral examples
- tokens and theming guidance

## Common Failure Modes

### Docs that describe the happy path only

Consumers learn how to render the component, but not how it behaves under loading, error, composition, or accessibility edge cases.

### Examples without decision guidance

Teams see snippets but still cannot tell which component variant or pattern is correct for their use case.

### Migration guidance after the fact

Breaking or risky changes land before consumers get meaningful upgrade documentation.

## Review Questions

- can a new engineer use the system correctly without asking a senior?
- where do teams still guess instead of follow explicit guidance?
- what docs are missing for high-risk components or upgrades?
- what part of the system currently requires too much tribal knowledge to use safely?
