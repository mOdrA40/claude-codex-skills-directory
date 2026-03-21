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

## Streaming Heuristics

### Structured Streaming is still semantics work

Micro-batch execution can make streaming feel familiar, but correctness still depends on explicit decisions about watermarking, state growth, replay posture, and sink behavior.

### Exactly-once claims must stop at the real boundary

The system may offer strong semantics internally while external sinks, side effects, and downstream consumers remain vulnerable to duplicates or ambiguous replay.

### Simpler batch may be safer than pseudo-streaming

If the workload does not truly need low-latency updates, stream-style complexity may create more operational and semantic risk than value.

## Common Failure Modes

### Streaming by fashion

The team uses structured streaming where periodic batch would be easier to reason about, cheaper to operate, and more trustworthy.

### Stateful correctness assumed, not tested

The job appears healthy in happy-path runs, but replay, restart, and late-data behavior remain weakly understood.

### Sink semantics hand-waved

Internal processing looks strong while the downstream sink still turns retries or replay into semantic ambiguity.

## Principal Review Lens

- What business result becomes wrong if state or watermark policy is off?
- Are we using structured streaming where simpler batch would be safer?
- Which external side effect is least safe under replay?
- What one design change most improves streaming trustworthiness?
- Which streaming guarantee sounds strongest in docs but weakest in real downstream behavior?
