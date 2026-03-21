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

## Principal Review Lens

- Where can duplicates still happen despite Flink features?
- Which sink or side effect breaks first under replay?
- Are we overclaiming correctness to product teams?
- What design improvement most strengthens end-to-end guarantees?
