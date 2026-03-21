# Reliability and Operations (Prometheus)

## Operational Defaults

- Monitor scrape health, target churn, rule evaluation duration, WAL pressure, disk growth, and remote write health.
- Keep Prometheus upgrades boring and reversible.
- Test backup, restore, and configuration rollout paths before incidents demand them.
- Separate platform troubleshooting from application telemetry disputes.

## Run-the-System Thinking

- Dashboards for Prometheus itself are mandatory in serious environments.
- Configuration review should catch cardinality and alert regressions before rollout.
- SLOs for the observability system should exist if teams depend on it operationally.
- On-call needs clear boundaries for what to fix locally versus escalate centrally.

## Principal Review Lens

- Can the team detect a broken observability pipeline before product teams do?
- Which failure mode hides the most truth while looking superficially healthy?
- Is Prometheus treated as critical infrastructure or as a sidecar afterthought?
- What operational practice would most reduce future observability incidents?
