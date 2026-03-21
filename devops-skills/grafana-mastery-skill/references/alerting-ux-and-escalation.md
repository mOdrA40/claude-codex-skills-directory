# Alerting UX and Escalation

## Rules

- Alert views should help people understand urgency, blast radius, and likely first actions.
- Routing, contact points, silences, and escalation policy require ownership discipline.
- Alert grouping should reduce noise without merging unrelated failures.
- The UI should support operational clarity, not page volume theater.

## Common Problems

- Alerts routed by historical accident rather than service ownership.
- Overgrouped incidents hiding multiple root causes.
- Silence abuse turning temporary mitigation into blind spots.
- Alerts visible in Grafana but disconnected from runbooks and action paths.

## Principal Review Lens

- Which alerting workflow wastes the most operator time?
- Can the current grouping logic hide a multi-service incident?
- Are silences treated as surgical tools or as a cleanup habit?
- Does the alerting UX shorten MTTR or just centralize noise?
