# Issuer Architecture and Trust Boundaries (cert-manager)

## Rules

- Issuer scope should reflect trust, ownership, and blast radius.
- Cluster-wide issuers and namespace-scoped issuers have different governance needs.
- Not every certificate should share the same trust and automation path.
- PKI automation should remain understandable to platform and security teams.

## Practical Guidance

- Separate public internet cert issuance from internal service PKI where needed.
- Keep tenant, environment, and domain ownership explicit.
- Align issuer design with DNS ownership and certificate lifecycle support.
- Avoid centralizing trust in one configuration path without strong controls.

## Principal Review Lens

- Which issuer has the highest blast radius today?
- Are we collapsing different trust domains into one automation path?
- What issuer boundary is too broad to be safe?
- Which PKI design choice most deserves refactoring?
