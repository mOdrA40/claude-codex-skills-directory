# Reliability and Operations (Spark)

## Operational Defaults

- Monitor job failure rates, executor health, shuffle pressure, storage dependency health, and queueing behavior.
- Keep runtime and platform changes staged and reversible.
- Distinguish cluster issues from job-design issues quickly.
- Document fallback paths for business-critical jobs.

## Run-the-System Thinking

- Spark platforms deserve reliability thinking because failed data pipelines often become business incidents.
- Capacity planning must include skew, shuffle, metadata, and queueing pressure.
- On-call should know which jobs and tables are most critical.
- Predictable standards beat heroic per-job tuning.

## Principal Review Lens

- What signal predicts a bad platform day earliest?
- Which workload should be protected or isolated first?
- Can the team explain current platform cost and failure posture clearly?
- Are we operating Spark as an engineered platform or a compute pit?
