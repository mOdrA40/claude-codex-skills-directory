# Dynamic Secrets and Lease Workflows

## Rules

- Dynamic secrets are valuable when they materially reduce exposure and simplify rotation.
- Lease, renewal, and revocation behavior must be understood by service owners.
- Secret issuance should align with application failure tolerance.
- Do not adopt dynamic workflows where teams cannot operate the lifecycle safely.

## Design Guidance

- Match lease duration to operational reality and threat model.
- Test renewal failure and expiration behavior under load and during outages.
- Keep revocation workflows predictable and observable.
- Ensure downstream systems tolerate credential churn.

## Principal Review Lens

- What secret is static today that should be dynamic?
- Which app would fail hardest if renewal breaks?
- Are we reducing risk or adding operational fragility?
- What lease policy best balances security and resilience?
