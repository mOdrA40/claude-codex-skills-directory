# Reliability and Operations (dbt)

## Operational Defaults

- Monitor run failures, freshness issues, contract breaks, model cost hotspots, and lineage blast radius of changes.
- Keep package, macro, and model changes staged and reviewable.
- Distinguish source-data failures from transformation-code failures quickly.
- Document fallback and rollback for critical models.

## Run-the-System Thinking

- dbt becomes platform infrastructure once many business decisions rely on it.
- Capacity planning includes warehouse cost, run concurrency, and model sprawl.
- On-call or platform owners should know which data products are most critical.
- Trust comes from explicit ownership, contracts, and boring deployment practice.

## Principal Review Lens

- Which signal predicts a bad analytics-platform day earliest?
- What model or source should be isolated first during trouble?
- Can the team explain current trust and cost posture clearly?
- Are we operating dbt as a governed platform or a SQL growth hack?
