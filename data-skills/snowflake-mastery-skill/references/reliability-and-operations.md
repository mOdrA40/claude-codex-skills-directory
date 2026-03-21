# Reliability and Operations (Snowflake)

## Operational Defaults

- Monitor warehouse saturation, query failures, security changes, sharing issues, and cost spikes.
- Keep privilege, masking, and warehouse policy changes reviewable and reversible.
- Distinguish platform issues from bad SQL or poor workload isolation quickly.
- Document emergency controls for runaway cost or broken access.

## Run-the-System Thinking

- Data platforms need reliability standards around critical pipelines and shared products.
- Cost governance is part of operational reliability when overruns threaten platform viability.
- On-call or platform owners should know which domains and warehouses are most business-critical.
- Trust comes from predictable governance and transparent ownership.

## Principal Review Lens

- Which signal predicts a bad platform day earliest?
- What workload or access change should be isolated first during trouble?
- Can the team explain current platform cost and access posture clearly?
- Are we operating a governed data platform or a convenient SQL mall?
