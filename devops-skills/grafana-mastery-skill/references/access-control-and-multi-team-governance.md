# Access Control and Multi-Team Governance

## Rules

- Folder structure, teams, roles, and permissions should reflect real ownership.
- Shared Grafana instances need policy before they need more dashboards.
- Sensitive data sources and executive dashboards require tighter review.
- Governance should preserve autonomy without creating observability anarchy.

## Governance Model

- Define who can create folders, provision data sources, and alter alerting routes.
- Separate platform-level dashboards from service-team dashboards.
- Track dashboard owners and stale content.
- Make deletion of unused dashboards part of maintenance.

## Principal Review Lens

- Which team has too much write power right now?
- Where is stale or ownerless content accumulating?
- Can one bad actor or one bad dashboard create wide operational confusion?
- Is governance reducing cognitive load or merely adding bureaucracy?
