# Reliability and Operations (Grafana)

## Operational Defaults

- Monitor datasource health, auth failures, dashboard load latency, alerting pipeline health, and plugin sprawl.
- Keep plugin usage controlled and audited.
- Backup and restore Grafana state if it is operationally critical.
- Treat Grafana outages separately from backend observability outages, even when they overlap.

## Run-the-System Thinking

- Users should know which Grafana instance is authoritative in multi-environment setups.
- Version upgrades should be staged and reversible.
- Authentication, SSO, and org/team sync are part of platform reliability.
- The observability portal itself needs SLOs if many teams depend on it.

## Principal Review Lens

- What makes Grafana unusable fastest during a big incident?
- Are we one plugin, auth issue, or datasource outage away from dashboard blindness?
- Can the team restore critical dashboards quickly if the instance fails?
- Which operational habit would most improve trust in Grafana?
