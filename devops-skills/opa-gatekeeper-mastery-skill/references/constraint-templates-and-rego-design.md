# Constraint Templates and Rego Design

## Rules

- Rego and constraint templates should remain readable, testable, and semantically narrow.
- One template should express a coherent class of policy.
- Template reuse should reduce duplication without hiding risk semantics.
- Policy code is production code and deserves review discipline.

## Practical Guidance

- Keep parameter surfaces constrained and meaningful.
- Write rules that humans can explain during incidents.
- Test edge cases and resource variations explicitly.
- Avoid policy logic that depends on fragile assumptions or unclear field shapes.

## Principal Review Lens

- Which template is hardest to understand or review today?
- Are we using abstraction to improve scale or to hide complexity?
- What policy code most deserves stronger tests?
- Which simplification would improve policy trust fastest?
