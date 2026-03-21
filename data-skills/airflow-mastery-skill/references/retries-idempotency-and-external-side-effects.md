# Retries, Idempotency, and External Side Effects

## Rules

- Retries are part of normal operation and must be accounted for in task design.
- Orchestration systems amplify side-effect mistakes if idempotency is weak.
- External state changes should be modeled carefully.
- Failure recovery should preserve correctness, not just rerun habit.

## Practical Guidance

- Make task side effects explicit and bounded.
- Design sinks and downstream systems for duplicate or partial execution scenarios.
- Distinguish transient retry paths from human escalation conditions.
- Test retry behavior under realistic external failures.

## Principal Review Lens

- Which task is least safe to retry today?
- Are we confusing orchestration success with business correctness?
- What external side effect is most likely to duplicate or corrupt?
- Which redesign most improves replay safety?
