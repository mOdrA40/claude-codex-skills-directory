# Navigation and Screen Flow Discipline

## Rules

- navigation structure should express product flow, not component convenience
- avoid screens that own too many unrelated async concerns
- pass stable route params, not giant objects
- model deep links and auth-gated flows explicitly

## Review Questions

- is navigation state recoverable on app resume?
- are screen responsibilities coherent?
- can deep links land safely in partial app state?
