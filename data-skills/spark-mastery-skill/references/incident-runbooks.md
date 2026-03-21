# Incident Runbooks (Spark)

## Cover at Minimum

- Shuffle/skew meltdown.
- Executor or memory failure wave.
- Storage or table-format dependency issue.
- Streaming state or replay incident.
- Queue starvation or multi-tenant overload.
- Critical job failure blocking business workflows.

## Incident Heuristics

### Classify by dominant cost center

Operators should quickly identify whether the main pain is:

- shuffle/skew
- memory / executor behavior
- storage or table-format dependency
- queue / tenancy contention
- stateful streaming correctness or replay behavior

### Protect critical pipelines before cluster elegance

Targeted throttling, queue separation, or temporary rollback is often better than global cluster tuning while business-critical jobs remain blocked.

### Recovery must include correctness, not just job completion

A Spark platform is not healthy again if jobs run but outputs remain stale, partial, or semantically untrustworthy.

## Response Rules

- Restore business-critical pipelines before perfect cluster efficiency.
- Prefer targeted throttling, queue isolation, or rollback over broad panic changes.
- Preserve stage, executor, and storage evidence for RCA.
- Communicate clearly about data freshness, correctness, and recovery windows.

## Common Failure Modes

### Cluster health without pipeline trust

The cluster stabilizes technically, but downstream users still cannot trust freshness or correctness.

### Generic compute response to data-layout problem

The team reacts as if the issue is raw compute shortage when the real problem is skew, shuffle shape, or storage interaction.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider platform instability?
- What proves the platform is healthy again?
- Are runbooks aligned with real Spark failure patterns?
- Which Spark incident should have a clearer safe-first degraded-mode path?
