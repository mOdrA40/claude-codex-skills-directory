# Data Sources and Correlation

## Rules

- Grafana should connect metrics, logs, traces, profiles, and events around the same service model.
- Each data source must have clear ownership, retention expectations, and use cases.
- Correlation should reduce time-to-insight, not add another maze of tabs.
- Source-specific limitations should be visible to users.

## Design Guidance

- Standardize service, environment, region, and tenant dimensions where possible.
- Link panels to logs and traces using stable identifiers and context labels.
- Make known gaps explicit: sampling, delayed ingest, retention mismatch, or incomplete coverage.
- Avoid presenting heterogeneous data as if all backends have equal truth quality.

## Principal Review Lens

- Can an operator move from symptom to evidence across data types quickly?
- Which correlation path is most brittle today?
- Are teams overtrusting one backend because Grafana makes it look unified?
- What naming inconsistency is hurting cross-tool debugging most?
