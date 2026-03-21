---
name: keycloak-principal-engineer
description: |
  Principal/Senior-level Keycloak playbook for identity architecture, realms and clients, federation, authorization, operational security, and operating IAM platforms at scale.
  Use when: designing IAM platforms, reviewing SSO and OAuth/OIDC flows, operating Keycloak clusters, or governing identity for many teams and applications.
---

# Keycloak Mastery (Senior → Principal)

## Operate

- Start from identity boundaries, trust flows, and blast radius of auth failures.
- Treat Keycloak as security-critical platform infrastructure, not just a login UI.
- Prefer explicit realm, client, role, and identity-provider boundaries.
- Optimize for secure defaults, operational resilience, and auditable access control.

## Default Standards

- Realm and client boundaries must reflect ownership and risk.
- Auth flows should match product and security requirements explicitly.
- Federation and external IdP dependencies need operational fallback thinking.
- Session, token, and credential lifecycle should be deliberate.
- Administrative access and realm changes require strong governance.

## References

- Realm, client, and tenant architecture: [references/realm-client-and-tenant-architecture.md](references/realm-client-and-tenant-architecture.md)
- OAuth, OIDC, SAML, and auth flow design: [references/oauth-oidc-saml-and-auth-flow-design.md](references/oauth-oidc-saml-and-auth-flow-design.md)
- Identity federation and external IdP strategy: [references/identity-federation-and-external-idp-strategy.md](references/identity-federation-and-external-idp-strategy.md)
- Roles, groups, authorization, and policy design: [references/roles-groups-authorization-and-policy-design.md](references/roles-groups-authorization-and-policy-design.md)
- Sessions, tokens, and lifecycle security: [references/sessions-tokens-and-lifecycle-security.md](references/sessions-tokens-and-lifecycle-security.md)
- Governance, admin security, and multi-team operations: [references/governance-admin-security-and-multi-team-operations.md](references/governance-admin-security-and-multi-team-operations.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
