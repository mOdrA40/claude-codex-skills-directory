# Reliability and Operations (Keycloak)

## Operational Defaults

- Monitor login failures, admin events, IdP latency, session anomalies, token issuance, and DB dependency health.
- Keep upgrades, realm changes, and auth-flow changes staged and reversible.
- Distinguish platform IAM issues from application integration issues quickly.
- Document safe fallback and break-glass access paths.

## Run-the-System Thinking

- Identity platforms deserve strong SLO thinking because auth failures quickly become business outages.
- Federation and DB dependencies are part of IAM availability design.
- On-call should know which realms, clients, and IdPs are most business-critical.
- Operational trust comes from boring, auditable identity workflows.

## Principal Review Lens

- Which failure blocks the most users fastest?
- Can the team recover auth safely without widening security risk?
- What operational habit most improves trust in Keycloak?
- Are we running identity as a platform or as a fragile admin console?
