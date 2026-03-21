# Incident Runbooks (cert-manager)

## Cover at Minimum

- Renewal failure surge.
- DNS or HTTP challenge outage.
- Issuer/CA dependency incident.
- Impending certificate expiry.
- Secret or key access mistake.
- High-blast-radius wildcard or shared-issuer issue.

## Response Rules

- Restore trust for critical endpoints before cleaning PKI elegance.
- Prefer targeted issuer or cert mitigation over broad risky edits.
- Preserve challenge, issuer, and secret evidence for RCA.
- Communicate clearly about trust state, expiry risk, and fallback paths.

## Principal Review Lens

- Can responders stop critical expiry blast radius quickly?
- Which emergency action most risks broader trust damage?
- What proves certificate automation is healthy again?
- Are runbooks usable under real expiry pressure?
