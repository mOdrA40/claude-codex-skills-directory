# Event Time, Watermarks, and Lateness Correctness

## Rules

- Event-time correctness must be based on real source behavior, not idealized ordering.
- Watermarks are part of business correctness, not just runtime tuning.
- Lateness handling should be explicit and testable.
- Product owners should understand when results may be late, revised, or dropped.

## Practical Guidance

- Measure source lateness distributions and clock behavior.
- Design windows and triggers around actual business semantics.
- Make retractions, updates, and late-data policy explicit downstream.
- Validate correctness under delayed and out-of-order replay scenarios.

## Correctness Heuristics

### Watermarks encode trust assumptions

A watermark strategy is a statement about how much lateness and disorder the business is willing to tolerate before results are considered complete enough.

### Downstream consumers must understand revision behavior

If late data can update, retract, or correct previous results, downstream systems and stakeholders need to know that explicitly.

## Common Failure Modes

### Happy-path watermarking

The system looks correct in demos and low-lateness tests, but real sources violate the assumptions and downstream results become misleading.

### Hidden correction semantics

Late or corrected events change aggregates later, but consumers were led to believe earlier outputs were final.

## Principal Review Lens

- What business result becomes wrong if watermark policy is off?
- Are we hiding late-data risk behind happy-path demos?
- Which source has the weakest event-time trustworthiness?
- What correctness tradeoff should be made explicit to stakeholders?
- What downstream contract becomes dangerous if revision behavior is misunderstood?
