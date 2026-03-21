# Incident Runbooks (Spark)

## Cover at Minimum

- Shuffle/skew meltdown.
- Executor or memory failure wave.
- Storage or table-format dependency issue.
- Streaming state or replay incident.
- Queue starvation or multi-tenant overload.
- Critical job failure blocking business workflows.

## Response Rules

- Restore business-critical pipelines before perfect cluster efficiency.
- Prefer targeted throttling, queue isolation, or rollback over broad panic changes.
- Preserve stage, executor, and storage evidence for RCA.
- Communicate clearly about data freshness, correctness, and recovery windows.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider platform instability?
- What proves the platform is healthy again?
- Are runbooks aligned with real Spark failure patterns?
