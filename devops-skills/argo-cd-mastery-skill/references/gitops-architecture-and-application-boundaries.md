# GitOps Architecture and Application Boundaries (Argo CD)

## Rules

- Application boundaries should reflect ownership, blast radius, and rollback semantics.
- Git should remain the authoritative desired state, not merely a suggestion.
- Avoid app definitions that collapse unrelated workloads into one risky sync unit.
- Repository structure and app boundaries should be understandable to operators.

## Design Guidance

- Separate platform, shared services, and application delivery concerns when they diverge operationally.
- Keep environment structure explicit and reviewable.
- Align repo layout with who approves and who supports changes.
- Prefer boring hierarchy over clever GitOps nesting.

## Principal Review Lens

- Which Argo app has the highest hidden blast radius?
- Are app boundaries aligned with real ownership or historical convenience?
- Can operators explain what one sync will actually touch?
- What split would most improve rollback safety?
