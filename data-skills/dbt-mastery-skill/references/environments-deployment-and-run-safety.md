# Environments, Deployment, and Run Safety

## Rules

- dbt deployment should reflect environment trust, cost, and blast radius.
- Runs must be reviewable in terms of downstream effect and warehouse cost.
- Backfills, full-refreshes, and stateful model evolution require stronger control.
- CI and production dbt workflows should be predictable and explainable.

## Practical Guidance

- Separate dev, CI, and prod behavior intentionally.
- Track high-risk operations such as full-refresh and large model invalidation.
- Keep rollback and emergency containment paths visible.
- Align run cadence with source freshness and consumer expectation.

## Principal Review Lens

- Which dbt operation has the highest blast radius today?
- Can the team explain what a production run will impact?
- Are environment differences helping safety or hiding drift?
- What change most improves run safety?
