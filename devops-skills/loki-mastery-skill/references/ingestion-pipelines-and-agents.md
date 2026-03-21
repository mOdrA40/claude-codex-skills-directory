# Ingestion Pipelines and Agents

## Rules

- Agent and pipeline design should reflect source diversity, parsing needs, and trust boundaries.
- Parsing should add value without turning ingestion into brittle ETL.
- Backpressure and burst behavior are part of system design.
- Log enrichment should be intentional and cost-aware.

## Design Guidance

- Keep source-specific parsing logic reviewable.
- Distinguish platform-generated logs from app logs where ownership differs.
- Watch drops, retries, parse failures, and ingest lag.
- Preserve enough raw truth to debug parsing errors later.

## Principal Review Lens

- Which agent or parser is most likely to fail silently?
- Are we over-processing logs during ingestion?
- What source would create most pain if it became noisy suddenly?
- Can operators explain the ingest path end-to-end quickly?
