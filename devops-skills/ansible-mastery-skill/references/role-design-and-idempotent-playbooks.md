# Role Design and Idempotent Playbooks

## Rules

- Roles should encode repeatable operational intent with clear inputs and safe defaults.
- Idempotency must hold under normal reruns and partial failure recovery.
- Tasks should be readable, scannable, and grouped by meaningful concern.
- Handlers, conditionals, and templates should remain understandable under incident pressure.

## Design Guidance

- Favor small, composable roles over sprawling monoliths when ownership differs.
- Keep shell usage rare and justified.
- Make state changes explicit and test rerun behavior.
- Reduce side effects hidden behind templates or conditionals.

## Principal Review Lens

- Can the team rerun this safely after partial failure?
- Which task is least idempotent in practice?
- Are we hiding imperative scripting behind YAML aesthetics?
- What role boundary should be refactored next?
