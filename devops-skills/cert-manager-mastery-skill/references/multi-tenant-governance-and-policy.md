# Multi-Tenant Governance and Policy

## Rules

- Shared certificate automation needs policy over issuers, domains, namespaces, and secret handling.
- Tenants should not be able to request certs outside their intended trust boundaries.
- Governance should focus on blast radius and supportability.
- Domain ownership and tenant authority must be explicit.

## Practical Guidance

- Standardize which issuers and certificate patterns tenants may use.
- Track owners of critical domains and wildcard certs.
- Keep exception flows explicit and auditable.
- Separate self-service convenience from high-risk trust domains.

## Principal Review Lens

- Which tenant can create the largest PKI incident today?
- Are domain and issuer controls enforceable or just documented?
- What policy gap most threatens certificate trust?
- Which governance rule most improves shared-platform safety?
