---
name: vault-principal-engineer
description: |
  Principal/Senior-level Vault playbook for secrets architecture, authn/authz, dynamic secrets, PKI, tenancy, and operating secure secret-management platforms in production.
  Use when: designing secret platforms, reviewing auth methods and policies, operating Vault clusters, or hardening access to sensitive systems.
---

# Vault Mastery (Senior → Principal)

## Operate

- Start from trust boundaries, blast radius, and secret lifecycle.
- Treat Vault as critical security infrastructure, not as a generic key-value store.
- Prefer explicit auth, policy, and tenancy boundaries.
- Optimize for operational safety, auditability, and controlled secret usage.

## Default Standards

- Auth methods and policies must reflect real identity boundaries.
- Dynamic secrets should be used where they materially reduce risk.
- Secret access should be time-bounded, least-privilege, and auditable.
- Recovery and seal/unseal posture must be operationally clear.
- Platform ownership and tenant boundaries should be explicit.

## References

- Secrets architecture and trust boundaries: [references/secrets-architecture-and-trust-boundaries.md](references/secrets-architecture-and-trust-boundaries.md)
- Auth methods, identity, and policy design: [references/auth-methods-identity-and-policy-design.md](references/auth-methods-identity-and-policy-design.md)
- Dynamic secrets and lease workflows: [references/dynamic-secrets-and-lease-workflows.md](references/dynamic-secrets-and-lease-workflows.md)
- PKI, certificates, and crypto operations: [references/pki-certificates-and-crypto-operations.md](references/pki-certificates-and-crypto-operations.md)
- Multi-tenant governance and access control: [references/multi-tenant-governance-and-access-control.md](references/multi-tenant-governance-and-access-control.md)
- Auditability, compliance, and secret lifecycle: [references/auditability-compliance-and-secret-lifecycle.md](references/auditability-compliance-and-secret-lifecycle.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
