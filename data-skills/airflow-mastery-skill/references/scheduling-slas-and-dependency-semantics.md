# Scheduling, SLAs, and Dependency Semantics

## Rules

- Schedule design should reflect data readiness, business SLA, and downstream expectations.
- Cron syntax is not workflow architecture.
- Backfill, catchup, and dependency rules must be operationally explicit.
- Airflow timing semantics should not surprise stakeholders.

## Practical Guidance

- Align schedule intervals with source-system readiness and business need.
- Make SLA expectations visible and actionable.
- Document how late upstream data affects downstream DAGs.
- Review cross-DAG dependencies and sensor use carefully.

## Scheduling Heuristics

### A schedule is a trust contract

Cron frequency alone says very little. What matters is whether the schedule aligns with source readiness, business expectations, and the truth users infer from data freshness.

### Dependency semantics should survive late data and backfills

Teams should know how sensors, external dependencies, catchup behavior, and manual backfills interact when the platform is stressed.

### SLAs need operational consequences

An SLA is useful only when breach detection, escalation, and business interpretation are explicit enough to guide action.

## Common Failure Modes

### Cron theater

The DAG runs on time according to the scheduler, but the data is still not ready or not trustworthy for the promised consumer use.

### Dependency fragility hidden by happy-path timing

Cross-DAG or sensor-heavy designs look acceptable until one upstream delay triggers widespread waiting, retries, or freshness confusion.

### Backfill semantics surprise

The business assumes one behavior during catchup or rerun, but the actual Airflow semantics produce a very different freshness or dependency outcome.

## Principal Review Lens

- Which DAG schedule is most detached from real data readiness?
- Are we using dependency constructs that create fragile orchestration?
- What backfill behavior would surprise the business most?
- Which schedule or SLA policy needs redesign first?
- Which timing assumption is currently least safe under delayed upstreams or reruns?
