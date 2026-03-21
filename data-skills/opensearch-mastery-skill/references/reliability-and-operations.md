# Reliability and Operations (OpenSearch)

## Operational Defaults

- Monitor heap, GC, indexing throughput, query latency, thread pools, shard health, and cluster state changes.
- Keep upgrades, template changes, and reindex workflows staged and reversible.
- Separate product-search incidents from observability-ingest incidents quickly.
- Test snapshots and restore workflows before trusting them.

## Run-the-System Thinking

- Different clusters may deserve different tuning and SLOs.
- Capacity planning must include node loss, reindex, and ingest spikes.
- On-call should know which indices and workloads are most critical.
- Operational simplicity often beats endless flexibility.

## Principal Review Lens

- What signal predicts cluster trouble earliest?
- Which index or workload would you isolate first during stress?
- Can the team explain node-role and topology choices clearly?
- Are we running a search platform or a pile of defaults?
