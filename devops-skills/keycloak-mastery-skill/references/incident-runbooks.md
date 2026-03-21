# Incident Runbooks (Keycloak)

## Cover at Minimum

- Login failure surge.
- External IdP outage.
- Bad realm/client config rollout.
- Token/session anomaly.
- Admin lockout or privilege mistake.
- Database or backend dependency issue.

## Response Rules

- Restore safe access to critical users and systems first.
- Prefer targeted rollback over wide identity surgery.
- Preserve auth logs, admin events, and IdP evidence for RCA.
- Communicate clearly about availability versus trust/security state.

## Principal Review Lens

- Can responders reduce blast radius without creating security debt?
- Which emergency action most risks widening identity failure?
- What proves the IAM platform is healthy again?
- Are runbooks realistic for multi-team auth incidents?
