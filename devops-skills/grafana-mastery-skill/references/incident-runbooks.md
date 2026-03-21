# Incident Runbooks (Grafana)

## Cover at Minimum

- Datasource-wide failures.
- Broken SSO or permission sync.
- Bad dashboard rollout.
- Alerting pipeline failures.
- Plugin or upgrade regressions.
- Correlation workflows breaking during active incidents.

## Response Rules

- Restore operator visibility before polishing dashboards.
- Prefer authoritative, minimal dashboards over complex broken ones.
- Preserve evidence for root cause analysis.
- Communicate clearly when Grafana is degraded but backends remain healthy.

## Principal Review Lens

- Can teams still triage if Grafana degrades?
- What workaround is safest when the main portal is unavailable?
- Which action fixes visibility fastest without corrupting state?
- What proves Grafana is truly healthy again, not just reachable?
