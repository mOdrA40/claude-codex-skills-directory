# Transactions and Multi-Region (CockroachDB)

## Transaction Rules

- Expect restart errors and code for them.
- Keep write sets minimal.
- Avoid long-lived transactions across user think time.
- Make idempotency explicit on externally visible side effects.

## Transaction Heuristics

### Treat retries as part of correctness design

If a transaction can restart, side effects around it must remain safe and understandable.

### Keep business invariants local when possible

The more cross-region coordination a transaction needs, the more expensive strong correctness becomes in latency and failure handling.

## Multi-Region Rules

- Put data near the writers/readers that matter most.
- Measure latency by region, not only cluster averages.
- Validate failover and degraded-region behavior with realistic tests.

## Multi-Region Design Questions

- which writes are region-local vs globally coordinated?
- which user journey pays the latency tax for strong guarantees?
- what happens when one region is slow rather than fully down?
- what business path becomes misleading if retries keep succeeding slowly?

## Common Failure Modes

### Logical correctness, poor operability

The design is technically correct but creates user-facing latency or retry behavior that support and application teams cannot reason about.

### Region averages hiding bad path cost

The cluster looks healthy overall while one region or one write path pays a disproportionate consistency penalty.

## Principal Review Lens

- What is the cost of strong consistency here?
- Which business flow breaks first during regional impairment?
- Is retry logic bounded and observable?
- What transaction should be redesigned before the workload grows further?
