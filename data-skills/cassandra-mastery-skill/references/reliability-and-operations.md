# Reliability and Operations (Cassandra)

## Operational Defaults

- Monitor latency, dropped mutations, repair health, compaction, tombstones, disk pressure, and node state.
- Keep topology changes and maintenance staged and reversible.
- Distinguish cluster-wide problems from table-specific or tenant-specific issues quickly.
- Document safe actions for node loss, hotspots, and streaming pressure.

## Run-the-System Thinking

- Operational trust comes from repair discipline and predictable topology management.
- Capacity headroom should include node loss and heavy maintenance events.
- On-call should know which workloads are most critical and most dangerous.
- Simplicity in schema and topology beats theoretical elegance.

## Principal Review Lens

- What signal predicts a bad cluster day earliest?
- Which workload would you isolate first during severe stress?
- Can the team explain current correctness and repair posture clearly?
- Are we operating Cassandra intentionally or fighting it reactively?
