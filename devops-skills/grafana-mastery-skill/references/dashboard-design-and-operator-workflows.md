# Dashboard Design and Operator Workflows (Grafana)

## Rules

- Every dashboard should answer a specific class of operational questions.
- Start with service health, user pain, dependencies, saturation, and recent change context.
- The first screen should help triage, not impress visitors.
- Avoid giant dashboards that require scrolling through chaos during incidents.

## Useful Dashboard Structure

- Summary row for SLO, errors, latency, traffic, saturation.
- Dependency row for databases, queues, caches, ingress, and downstreams.
- Change context row for deploys, feature flags, or config rollouts.
- Drilldown row for outliers, tenants, regions, or pods.

## Principal Review Lens

- Can an on-call engineer use this dashboard in under two minutes?
- Which panel exists only because nobody deleted it?
- Does the dashboard reflect how incidents actually unfold?
- What decision becomes faster because this dashboard exists?
