# Reliability and Operations (Airflow)

## Operational Defaults

- Monitor scheduler health, executor/worker capacity, metadata DB, DAG parse performance, and SLA miss patterns.
- Keep DAG and platform changes staged and reversible.
- Distinguish platform-wide orchestration issues from one-workflow problems quickly.
- Document fallback paths for business-critical workflows.

## Run-the-System Thinking

- Airflow becomes critical infrastructure once many business processes depend on it.
- Capacity planning includes scheduler throughput, worker fleet, and dependency reliability.
- On-call should know which DAGs matter most and which can degrade safely.
- Operational trust comes from boring DAGs and disciplined platform governance.

## Principal Review Lens

- Which Airflow failure blocks the most business value fastest?
- Can the team recover scheduler or metadata problems quickly?
- What operational habit most improves trust in the orchestration platform?
- Are we operating Airflow as a workflow control plane or a dumping ground?
