# Delivery Semantics and External Side Effects

## Rules

- Exactly-once claims must be evaluated across the whole boundary, not just inside Flink.
- External side effects require idempotency, transactional design, or explicit compensation.
- Replay and retries are normal operational realities.
- Sink semantics should match business correctness needs.

## Practical Guidance

- Distinguish internal state correctness from external effect correctness.
- Design side-effect systems for duplicates and reordering where necessary.
- Test failure behavior during checkpoint and sink interruptions.
- Make operator and stakeholder expectations explicit.

## Semantics Heuristics

### End-to-end guarantees end at the weakest boundary

Flink may preserve strong internal semantics while an external sink, API, or side-effect system still turns retries and replay into ambiguity.

### External effects need their own safety model

The right design may require idempotency keys, transactional boundaries, deduplication tables, compensating actions, or explicit acceptance of duplicate risk.

### Operators need to know what replay means

During incidents, responders should be able to explain whether replay creates more load, duplicate effects, business corrections, or all three.

## Common Failure Modes

### Internal correctness overclaimed as workflow correctness

The platform describes semantics confidently while downstream side effects remain only partially protected.

### Sink interruption ambiguity

The pipeline recovers eventually, but nobody can confidently explain which external effects happened once, twice, or not at all.

### Replay without business-language guardrails

Technical teams understand replay mechanics, but stakeholders do not understand what it means for orders, alerts, balances, or user-facing state.

## Principal Review Lens

- Where can duplicates still happen despite Flink features?
- Which sink or side effect breaks first under replay?
- Are we overclaiming correctness to product teams?
- What design improvement most strengthens end-to-end guarantees?
- Which external effect currently has the weakest idempotency or compensation story?
