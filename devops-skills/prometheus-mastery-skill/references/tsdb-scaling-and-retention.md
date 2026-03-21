# TSDB Scaling and Retention

## Rules

- Retention is a cost, performance, and incident-forensics decision.
- Cardinality, scrape interval, histogram usage, and rule volume determine TSDB pressure together.
- Local Prometheus should optimize for fast operational queries, not unlimited history.
- Retention growth should be forecast before disk pressure becomes an outage.

## Capacity Thinking

- Estimate series growth by team, service, label set, and rollout behavior.
- Model headroom for backfill, churn, target flapping, and failure bursts.
- Treat compaction and WAL behavior as operational concerns, not internals to ignore.
- Plan remote storage based on retrieval patterns, not only cost per GB.

## Principal Review Lens

- What causes storage growth fastest in this environment?
- Which dimension makes yesterday's safe design dangerous next quarter?
- Are we keeping long retention locally because it is useful or because no one decided otherwise?
- What fails first under 2x cardinality: disk, memory, query speed, or operator patience?
