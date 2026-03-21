# Reliability and Operations (ScyllaDB)

## Operational Defaults

- Monitor tail latency, hot shards, repair health, compaction, disk, and node state.
- Keep topology changes and maintenance staged and reversible.
- Distinguish cluster-wide pressure from one-tenant or one-table pain quickly.
- Document safe response paths for hotspots and node impairment.

## Run-the-System Thinking

- Operational excellence is inseparable from data-model quality.
- Capacity headroom should include maintenance and node loss.
- On-call should know which workloads matter most and which can be degraded.
- Low-latency claims are only credible with disciplined operations.

## Principal Review Lens

- What signal predicts cluster trouble earliest?
- Which workload should be isolated first during severe stress?
- Can the team explain current correctness and maintenance posture clearly?
- Are we operating ScyllaDB intentionally or surfing benchmarks?
