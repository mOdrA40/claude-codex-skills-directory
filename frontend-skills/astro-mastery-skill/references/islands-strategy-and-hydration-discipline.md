# Islands Strategy and Hydration Discipline

## Principle

Astro performance wins come from disciplined hydration. If every interactive need becomes an island without rules, complexity grows and the original delivery advantage shrinks.

## Common Failure Modes

- too many islands with fuzzy ownership
- hydration added where simple HTML behavior would suffice
- interactive state split awkwardly across multiple isolated islands

## Practical Heuristics

### Favor zero-JS by default

If a page or region can satisfy the product need with HTML, CSS, and server-rendered content, make that the baseline rather than the fallback.

### Hydrate by user value, not by convenience

Interactive surfaces that drive actual user workflows deserve hydration sooner than decorative or secondary enhancements.

### Keep ownership local to the island

If multiple islands depend on tightly shared state, the page architecture may be fighting the framework model.

## Review Questions

- which parts of the page truly need interactivity?
- what hydration cost is acceptable for this route?
- where is interactivity being added by habit rather than product need?
- which island would be hardest to reason about during a production incident?
