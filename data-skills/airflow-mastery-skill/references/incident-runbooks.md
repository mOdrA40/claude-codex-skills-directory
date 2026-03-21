# Incident Runbooks (Airflow)

## Cover at Minimum

- Scheduler degradation.
- Metadata DB issue.
- Queue starvation or worker exhaustion.
- Critical DAG failure wave.
- Bad DAG deployment.
- Backfill or retry storm.

## Incident Heuristics

### Separate platform failure from DAG failure

Operators should quickly determine whether the incident is:

- scheduler/platform level
- executor/worker capacity level
- metadata database level
- one DAG family or dependency contract level

### Protect critical orchestration first

The right first move is often to isolate or pause non-critical workflows so that business-critical DAGs can recover predictably.

### Recovery must include freshness truth

An Airflow incident is not resolved just because tasks start moving again. Teams must know what data freshness and dependency state users can trust.

## Response Rules

- Restore business-critical orchestration before broad platform cleanup.
- Prefer targeted DAG isolation and rollback over panic-wide changes.
- Preserve run metadata, logs, and dependency evidence for RCA.
- Communicate clearly about data freshness, workflow status, and recovery timelines.

## Common Failure Modes

### DAG wave mistaken for platform collapse

One bad DAG deployment or dependency contract failure creates widespread pain, but teams initially treat it like generic platform instability.

### Partial scheduler recovery mistaken for business recovery

The platform starts executing again, but critical freshness or dependency gaps remain unresolved.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider scheduling instability?
- What proves the platform is healthy again?
- Are runbooks aligned with real Airflow failure modes?
- Which Airflow incident still relies too much on tribal operator judgement?
