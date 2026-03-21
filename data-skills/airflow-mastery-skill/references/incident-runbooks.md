# Incident Runbooks (Airflow)

## Cover at Minimum

- Scheduler degradation.
- Metadata DB issue.
- Queue starvation or worker exhaustion.
- Critical DAG failure wave.
- Bad DAG deployment.
- Backfill or retry storm.

## Response Rules

- Restore business-critical orchestration before broad platform cleanup.
- Prefer targeted DAG isolation and rollback over panic-wide changes.
- Preserve run metadata, logs, and dependency evidence for RCA.
- Communicate clearly about data freshness, workflow status, and recovery timelines.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider scheduling instability?
- What proves the platform is healthy again?
- Are runbooks aligned with real Airflow failure modes?
