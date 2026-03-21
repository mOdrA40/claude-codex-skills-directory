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

## Principal Review Lens

- Which DAG schedule is most detached from real data readiness?
- Are we using dependency constructs that create fragile orchestration?
- What backfill behavior would surprise the business most?
- Which schedule or SLA policy needs redesign first?
