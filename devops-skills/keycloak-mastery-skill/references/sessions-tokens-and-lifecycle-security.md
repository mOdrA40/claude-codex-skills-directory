# Sessions, Tokens, and Lifecycle Security

## Rules

- Token lifetime and session policy should match risk, UX, and operational need.
- Refresh tokens, offline tokens, and long-lived sessions carry real blast radius.
- Revocation and logout behavior should be understood by service owners.
- Credential lifecycle is part of security operations, not just app configuration.

## Practical Guidance

- Keep token TTLs and refresh rules explicit by client type.
- Test revocation, logout propagation, and session expiry behavior.
- Monitor unusual token issuance or session anomalies.
- Treat admin and service credentials as especially high risk.

## Principal Review Lens

- Which token or session type is most dangerous if leaked?
- Are current TTL choices aligned with actual threat model?
- What lifecycle gap most threatens access hygiene?
- Can the team explain logout and revocation semantics clearly?
