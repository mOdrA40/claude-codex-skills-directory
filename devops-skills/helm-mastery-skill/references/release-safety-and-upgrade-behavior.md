# Release Safety and Upgrade Behavior

## Rules

- Upgrade behavior should be predictable, reversible, and well-understood.
- Chart changes that force immutable-field replacement or selector drift are high risk.
- Hooks and jobs should be used carefully and reviewed for failure semantics.
- Rollback posture matters as much as install success.

## Failure Modes

- A chart upgrade rendering valid YAML but unsafe rollout behavior.
- Changed defaults breaking existing consumers silently.
- Hooks causing partial success with unclear cleanup.
- Rollbacks failing because stateful side effects were not considered.

## Principal Review Lens

- What happens to a live workload if this chart is upgraded at peak traffic?
- Which rendered change is syntactically safe but operationally dangerous?
- Can the team rollback without manual cluster surgery?
- Are we treating release safety as first-class design input?
