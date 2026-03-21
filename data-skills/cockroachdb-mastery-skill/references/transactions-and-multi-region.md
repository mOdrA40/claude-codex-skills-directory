# Transactions and Multi-Region (CockroachDB)

## Transaction Rules

- Expect restart errors and code for them.
- Keep write sets minimal.
- Avoid long-lived transactions across user think time.
- Make idempotency explicit on externally visible side effects.

## Multi-Region Rules

- Put data near the writers/readers that matter most.
- Measure latency by region, not only cluster averages.
- Validate failover and degraded-region behavior with realistic tests.

## Principal Review Lens

- What is the cost of strong consistency here?
- Which business flow breaks first during regional impairment?
- Is retry logic bounded and observable?
