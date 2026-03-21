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

## Trust Heuristics

### Not all models deserve equal testing depth

The strongest testing and contract discipline should concentrate on models with the highest consumer dependency, financial impact, or executive visibility.

### Trust signals should be legible to consumers

If only the dbt team understands what a passing suite means, the trust system is too internal to guide the business well.

## Common Failure Modes

### Green tests, low trust

Generic tests pass, but consumers still do not know whether freshness, semantics, or ownership are reliable enough for decision-making.

### Contract theater

Contracts exist in docs or configs, but no one treats breaking them as an operational event with real ownership.

## Principal Review Lens

- Which model is most business-critical yet least well guarded?
- Are we testing meaning or merely shape?
- What trust signal would most improve stakeholder confidence?
- Which failure path deserves stronger contract definition?
- Which model looks “tested” but would still surprise consumers badly on failure?
