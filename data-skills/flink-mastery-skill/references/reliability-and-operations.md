# Reliability and Operations (Flink)

## Operational Defaults

- Monitor checkpoint health, backpressure, operator lag, state growth, sink errors, and restart behavior.
- Keep job upgrades and savepoint workflows staged and reversible.
- Distinguish source, Flink, and sink failures quickly.
- Document safe fallback for critical streaming pipelines.

## Run-the-System Thinking

- Stream platforms deserve SLOs if product correctness depends on them.
- Capacity planning should include bursts, late data, and recovery events.
- On-call should know which jobs are most critical and least resilient.
- Operational simplicity beats magical stream pipelines.

## Principal Review Lens

- What signal predicts a bad streaming day earliest?
- Which job should be protected first under pressure?
- Can the team explain current correctness and recovery posture clearly?
- Are we operating Flink intentionally or worshipping stateful complexity?
