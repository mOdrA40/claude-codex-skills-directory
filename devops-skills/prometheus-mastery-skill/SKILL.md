---
name: prometheus-principal-engineer
description: |
  Principal/Senior-level Prometheus playbook for metrics design, TSDB operations, alerting quality, scaling, federation, remote write, and production observability architecture.
  Use when: designing metrics strategy, reviewing alerting posture, operating Prometheus at scale, debugging scrape/query/cardinality issues, or building multi-cluster observability platforms.
---

# Prometheus Mastery (Senior → Principal)

## Operate

- Start from user pain and SLOs, not dashboard vanity.
- Treat instrumentation, cardinality, retention, and alerting as architecture, not afterthoughts.
- Separate what needs fast local query from what needs long retention.
- Prefer boring, explainable metrics over noisy telemetry sprawl.

## Default Standards

- Metric names and labels must encode stable semantics.
- Cardinality is a budget, not a surprise.
- Alerts must be actionable, low-noise, and tied to risk.
- Recording rules should reduce operator toil, not create hidden logic.
- Scrape posture, retention, and storage growth should be explicit.

## Review Lens

- What business or operational decision does this metric enable?
- Which labels can explode without warning?
- Which alerts page humans without shortening incidents?
- Can the team explain retention and remote write tradeoffs clearly?

## References

- Metrics design and instrumentation: [references/metrics-design-and-instrumentation.md](references/metrics-design-and-instrumentation.md)
- PromQL and recording rules: [references/promql-and-recording-rules.md](references/promql-and-recording-rules.md)
- Alerting strategy and fatigue: [references/alerting-strategy-and-fatigue.md](references/alerting-strategy-and-fatigue.md)
- TSDB scaling and retention: [references/tsdb-scaling-and-retention.md](references/tsdb-scaling-and-retention.md)
- Federation and remote write: [references/federation-and-remote-write.md](references/federation-and-remote-write.md)
- Service discovery and scrape architecture: [references/service-discovery-and-scrape-architecture.md](references/service-discovery-and-scrape-architecture.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
