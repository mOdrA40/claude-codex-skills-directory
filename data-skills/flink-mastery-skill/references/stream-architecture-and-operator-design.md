# Stream Architecture and Operator Design (Flink)

## Rules

- Operator design should reflect business semantics, fault tolerance, and state boundaries.
- Pipelines should be decomposed into understandable stages with explicit ownership.
- Avoid monolithic jobs that hide correctness and operational risk.
- Stateful and stateless operators deserve different review rigor.

## Practical Guidance

- Identify where partitioning, enrichment, joins, and sinks create complexity.
- Keep transformation logic aligned with event contracts.
- Make state boundaries and rescaling implications visible.
- Use composition that helps operators debug data flow and failure points.

## Principal Review Lens

- Which operator boundary hides the most risk today?
- Are we modeling data semantics clearly or just chaining transformations?
- What job split would most improve recoverability?
- Which stage becomes hardest to reason about during incident response?
