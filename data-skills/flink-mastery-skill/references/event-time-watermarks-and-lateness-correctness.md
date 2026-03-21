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

## Principal Review Lens

- What business result becomes wrong if watermark policy is off?
- Are we hiding late-data risk behind happy-path demos?
- Which source has the weakest event-time trustworthiness?
- What correctness tradeoff should be made explicit to stakeholders?
