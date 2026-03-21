# Workspaces, Params, Results, and Contract Design

## Rules

- Tekton task contracts should be explicit, stable, and reviewable.
- Workspaces and shared state should be minimized and justified.
- Params and results are API boundaries between tasks.
- Hidden assumptions about files, paths, and artifacts create fragile pipelines.

## Practical Guidance

- Standardize task interfaces for common platform patterns.
- Keep mutable shared workspaces constrained.
- Validate that outputs are meaningful to downstream tasks.
- Treat task contract changes like breaking API changes for consumers.

## Principal Review Lens

- Which task contract is least clear today?
- Are we using shared workspace because we must or because it is easy?
- What hidden artifact assumption most threatens reproducibility?
- Which contract deserves versioning discipline next?
