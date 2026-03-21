# Structured Streaming and Stateful Semantics

## Rules

- Stateful streaming correctness depends on explicit watermark, state, and sink semantics.
- Micro-batch convenience does not eliminate stream-processing tradeoffs.
- Exactly-once claims must be evaluated across the external boundary.
- Streaming design should reflect lateness, replay, and downstream impact clearly.

## Practical Guidance

- Track state growth, trigger timing, and sink behavior.
- Test restart and replay behavior, not just steady-state output.
- Distinguish low-latency requirements from simple periodic micro-batch jobs.
- Make correctness and lateness tradeoffs visible to stakeholders.

## Principal Review Lens

- What business result becomes wrong if state or watermark policy is off?
- Are we using structured streaming where simpler batch would be safer?
- Which external side effect is least safe under replay?
- What one design change most improves streaming trustworthiness?
