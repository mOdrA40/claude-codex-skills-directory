# Observability and Debugging (Airflow)

## Rules

- Observability should reveal scheduler health, task failures, queueing, retries, and dependency bottlenecks clearly.
- Debugging must distinguish DAG logic issues from platform capacity or external dependency failures.
- UI and logs should support real incident workflows.
- Monitoring should prioritize business-critical DAGs and timing semantics.

## Useful Signals

- Scheduler lag, queued tasks, worker saturation, SLA misses, retry storms, and metadata DB health.
- Correlate DAG failures with upstream systems and data freshness signals.
- Standardize dashboards for platform health and DAG criticality.
- Preserve enough run history for RCA.

## Principal Review Lens

- Can on-call localize a DAG failure quickly?
- Which missing signal most slows Airflow diagnosis today?
- Are teams over-attributing problems to Airflow itself?
- What observability change most reduces incident time?
