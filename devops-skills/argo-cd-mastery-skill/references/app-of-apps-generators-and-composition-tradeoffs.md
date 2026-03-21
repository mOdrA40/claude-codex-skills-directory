# App-of-Apps, Generators, and Composition Tradeoffs

## Rules

- Composition patterns should reduce toil without hiding ownership or blast radius.
- Generators and app-of-apps can help scale, but they can also create opaque control planes.
- Abstraction should preserve operational meaning.
- Generated desired state must remain reviewable by humans.

## Practical Guidance

- Use composition where it simplifies repeated, consistent patterns.
- Avoid nesting that makes change impact hard to predict.
- Track ownership of generator logic and shared templates.
- Keep failure diagnosis straightforward when generated apps drift or fail.

## Principal Review Lens

- Which composition layer is the least understandable today?
- Are we generating leverage or just generating YAML complexity?
- What shared template has the largest silent blast radius?
- Which abstraction should be flattened?
