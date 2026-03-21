# Testing, Contracts, and Trust Signals

## Rules

- Testing should support confidence in semantics, freshness, and model contracts.
- Generic tests alone do not equal trustworthy analytics.
- Contracts and freshness expectations should be visible to consumers.
- Test suites should reflect business-critical risk, not just easy coverage counts.

## Practical Guidance

- Prioritize tests around keys, relationships, freshness, and key business invariants.
- Use exposures, contracts, and documentation to signal trust levels.
- Track stale tests and flaky trust checks as operational debt.
- Keep severity and escalation paths aligned with consumer impact.

## Principal Review Lens

- Which model is most business-critical yet least well guarded?
- Are we testing meaning or merely shape?
- What trust signal would most improve stakeholder confidence?
- Which failure path deserves stronger contract definition?
