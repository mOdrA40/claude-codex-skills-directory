# Testing, Linting, and Diff Workflows

## Rules

- Rendering, linting, and diffing should be routine before chart changes merge.
- Tests should validate the combinations that actually matter in production.
- Visual manifest diffs help reviewers spot operational risk earlier.
- CI should catch breaking chart API changes and dangerous output drift.

## Useful Practices

- Test representative environment overlays, not only defaults.
- Validate schema or values contracts where possible.
- Compare rendered output across chart upgrades for critical workloads.
- Keep test fixtures aligned with real deployment patterns.

## Principal Review Lens

- Which production configuration is currently untested?
- Can the current diff workflow reveal selector, security, or ingress regressions?
- Are tests preventing incidents or merely satisfying process?
- What one extra validation would most improve release safety?
