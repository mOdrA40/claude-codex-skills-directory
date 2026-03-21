# Incident Runbooks (Vault)

## Cover at Minimum

- Auth or policy lockout.
- Secret backend or storage issue.
- Seal/unseal or quorum problem.
- Lease/renewal outage.
- PKI compromise or certificate incident.
- Audit pipeline failure.

## Response Rules

- Restore safe access to critical systems before broad cleanup.
- Protect audit evidence during emergency action.
- Prefer reversible changes and minimal blast radius during incident response.
- Communicate clearly about secret validity, access, and trust state.

## Principal Review Lens

- Can responders recover critical access without undermining security?
- Which emergency action risks permanent trust damage?
- What evidence proves Vault is truly healthy again?
- Are runbooks prepared for both security and availability stress?
