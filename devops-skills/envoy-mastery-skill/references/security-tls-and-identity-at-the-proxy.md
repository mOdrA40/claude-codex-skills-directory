# Security, TLS, and Identity at the Proxy

## Rules

- TLS termination and identity propagation should reflect trust boundaries intentionally.
- Certificates, validation context, and downstream trust decisions are operationally critical.
- Sensitive headers and client identity handling require disciplined policy.
- Security config should remain auditable and predictable.

## Practical Guidance

- Separate external and internal trust models clearly.
- Test certificate rotation and trust-bundle changes safely.
- Keep authn/authz responsibility boundaries explicit between proxy and application.
- Standardize secure defaults wherever possible.

## Principal Review Lens

- Which proxy path has the weakest real security posture?
- Can one cert or trust misconfiguration widen blast radius dangerously?
- Are security features consistent enough to trust across teams?
- What security assumption is least validated today?
