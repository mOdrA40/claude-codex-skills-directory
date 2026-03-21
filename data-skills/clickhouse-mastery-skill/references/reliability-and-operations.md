# Reliability and Operations (ClickHouse)

## Operational Defaults

- Monitor ingest lag, part counts, merge pressure, disk utilization, query latency, and replica health.
- Keep upgrades, schema changes, and backfills staged and reversible.
- Distinguish analytical user pain from platform maintenance noise.
- Document operator playbooks for replica repair, disk pressure, and runaway query control.

## Run-the-System Thinking

- Clusters serving dashboards and ad-hoc analysts need different guardrails than ingestion-heavy observability clusters.
- Capacity headroom should include node loss and replay scenarios.
- On-call should know which workloads to protect first during degradation.
- Operational simplicity beats maximal feature use in most teams.

## Principal Review Lens

- Which operational signal predicts a bad day earliest?
- What workload should be throttled or isolated first during stress?
- Can the team explain cluster behavior under backfill and peak query load?
- Are we operating ClickHouse intentionally or just enjoying fast benchmarks?
