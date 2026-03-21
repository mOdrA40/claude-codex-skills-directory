# Logging vs Search Clusters (Elasticsearch)

## Rules

- User-facing search and observability ingestion have different optimization goals.
- Mixing both on one cluster often creates conflicting priorities.
- Query SLAs and ingest SLAs should be separated explicitly.
- Retention and relevance needs should not fight each other silently.

## Workload Heuristics

### Search and logging solve different business problems

User search prioritizes relevance, freshness, and interactive latency. Logging and observability prioritize ingest resilience, retention economics, broad aggregation, and operator workflows. One cluster can host both only with explicit tradeoff ownership.

### Separation should follow recurring pain, not ideology

The right reason to split clusters is that one workload repeatedly harms the trust, latency, or cost posture of the other—not merely because best-practice slogans say so.

### Shared clusters need explicit winner rules

If logging and search stay together, the platform should know which workload gets priority when heap, storage, indexing, or latency pressure makes them compete.

## Common Failure Modes

### One-cluster convenience tax

The platform avoids separation to simplify operations, but the combined workload creates higher hidden cost and worse user experience over time.

### Search users paying logging debt

Heavy ingest and retention behavior slowly erode the responsiveness or stability expected by user-facing search.

### Logging expectations hidden inside search tuning

Cluster defaults are tuned for one dominant use case while the other suffers silently until a larger incident exposes the mismatch.

## Principal Review Lens

- Are we solving two incompatible workloads with one default setup?
- Which workload wins during resource contention?
- Would separation reduce operational tax materially?
- Which mixed-workload assumption is currently least governed explicitly?
