# Environment Strategy and Promotion

## Rules

- Environment design should match delivery flow, compliance needs, and ownership boundaries.
- Promotion should be explainable: what moves, when, and under whose approval.
- Workspace, directory, and repo strategies each have tradeoffs and should be chosen intentionally.
- Shared modules do not require shared state or shared deployment cadence.

## Common Mistakes

- Treating environments as copy-paste folders with unmanaged drift.
- Using workspaces where separate state and policy boundaries are clearer.
- Mixing rapid app iteration concerns with slow-moving foundation concerns.
- Making promotion logic tribal rather than codified.

## Principal Review Lens

- Does this environment model reduce or increase rollout risk?
- What kind of misconfiguration can leak between environments?
- Can the team prove production is derived from reviewed changes rather than console mutation?
- Which environment boundary is operationally fake today?
