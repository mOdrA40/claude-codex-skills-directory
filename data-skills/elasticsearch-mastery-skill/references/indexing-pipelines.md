# Indexing Pipelines (Elasticsearch)

## Rules

- Ingestion pipelines should be deterministic and observable.
- Enrichment and transforms add cost that must be justified.
- Keep ingest complexity proportional to business value.
- Guard against malformed or exploding documents.

## Principal Review Lens

- Which ingest processor dominates indexing cost?
- What bad document can poison the pipeline?
- Is transformation better done before Elasticsearch?
