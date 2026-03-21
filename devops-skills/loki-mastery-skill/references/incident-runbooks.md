# Incident Runbooks (Loki)

## Cover at Minimum

- Ingest failure or backlog.
- Bad label rollout causing cardinality pain.
- Query storm or broad-search overload.
- Object storage or index dependency incident.
- Tenant-specific abuse affecting shared platform.
- Retention or deletion misconfiguration.

## Response Rules

- Restore critical investigative capability before chasing perfect coverage.
- Prefer isolating noisy tenants and risky queries over global panic changes.
- Preserve evidence about drops, lag, and query offenders.
- Communicate clearly when logs are partial, delayed, or unreliable.

## Principal Review Lens

- Can responders regain useful log visibility quickly?
- Which emergency action most risks hiding needed evidence?
- What confirms the platform is truly healthy again?
- Are runbooks usable during real multi-team incidents?
