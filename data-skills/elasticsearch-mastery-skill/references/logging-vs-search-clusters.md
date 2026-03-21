# Logging vs Search Clusters (Elasticsearch)

## Rules

- User-facing search and observability ingestion have different optimization goals.
- Mixing both on one cluster often creates conflicting priorities.
- Query SLAs and ingest SLAs should be separated explicitly.
- Retention and relevance needs should not fight each other silently.

## Principal Review Lens

- Are we solving two incompatible workloads with one default setup?
- Which workload wins during resource contention?
- Would separation reduce operational tax materially?
