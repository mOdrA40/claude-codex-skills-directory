# Ingestion Pipelines and Retention

## Rules

- Ingest pipelines should be deterministic, observable, and bounded in complexity.
- Retention is a business, security, and cost decision.
- Logging-style ingestion and search-style ingestion should not share defaults carelessly.
- Bad documents and malformed mappings require explicit failure handling.

## Design Guidance

- Keep transformations intentional and justified.
- Align rollover, retention, and tiering with access patterns.
- Watch ingest burstiness, backpressure, and pipeline processor cost.
- Make deletion, archive, and retention ownership explicit.

## Principal Review Lens

- Which ingest processor creates the most hidden tax?
- Are we retaining data longer than its real value?
- What failure turns one bad source into cluster-wide pain?
- Can the team explain retention policy in business terms?
