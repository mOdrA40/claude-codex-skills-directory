# Job Architecture and Dataflow Design (Spark)

## Rules

- Spark job design should reflect business semantics, failure domains, and cost expectations.
- Explicit stages and dataflow boundaries improve tuning and debugging.
- Avoid monolithic jobs that hide shuffle and failure complexity.
- Batch and streaming designs should each be optimized for their own semantics.

## Practical Guidance

- Make inputs, transformations, aggregations, and outputs easy to reason about.
- Align orchestration and retry behavior with downstream effects.
- Keep expensive joins and wide dependencies visible in design review.
- Standardize patterns that improve readability and operability.

## Principal Review Lens

- Which dataflow boundary hides the most cost or failure risk?
- Are we building understandable jobs or giant notebooks in disguise?
- What split would most improve retry and support posture?
- Which stage becomes hardest to debug under pressure?
