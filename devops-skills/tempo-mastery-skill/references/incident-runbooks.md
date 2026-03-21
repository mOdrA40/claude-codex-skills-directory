# Incident Runbooks (Tempo)

## Cover at Minimum

- Ingest surge and dropped traces.
- Search degradation.
- Object storage dependency failure.
- Compaction or block lifecycle issue.
- Tenant-driven overload.
- Broken correlation paths from Grafana or exemplars.

## Response Rules

- Restore critical trace usefulness before tuning perfect retention or query behavior.
- Prefer targeted tenant or ingest control over broad risky changes.
- Preserve evidence about drops, search failures, and storage errors.
- Communicate clearly when traces are delayed, partial, or untrustworthy.

## Principal Review Lens

- Can responders regain trustworthy traces quickly?
- Which emergency action risks hiding or deleting needed evidence?
- What confirms the platform is healthy again end-to-end?
- Are runbooks realistic for observability-platform incidents?
