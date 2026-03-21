# Ingestion Pipelines and Late Data

## Rules

- Ingestion design must reflect source ordering, lateness, deduplication, and throughput expectations.
- ClickHouse can ingest extremely fast, but merge and part behavior still need respect.
- Late or corrected events require explicit semantics, not wishful append-only assumptions.
- Batch size, compression, and write concurrency influence system health together.

## Operational Guidance

- Distinguish streaming freshness goals from batch efficiency goals.
- Validate how retries and duplicates behave per table engine and dedup strategy.
- Model backfill and replay workflows before the first major correction event.
- Keep ingest-path observability tied to lag, part count, and merge pressure.

## Principal Review Lens

- What happens when data arrives late by hours or days?
- Which ingest retry pattern can silently create duplicates or merge pain?
- Are backfills operationally safe under normal production load?
- What source-system assumption is most dangerous to trust here?
