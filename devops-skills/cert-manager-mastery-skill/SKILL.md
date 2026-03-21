---
name: cert-manager-principal-engineer
description: |
  Principal/Senior-level cert-manager playbook for certificate lifecycle automation, issuer architecture, trust boundaries, DNS/HTTP challenge strategy, and operating PKI automation on Kubernetes safely.
  Use when: designing certificate automation, reviewing issuers and trust, operating cert-manager, or scaling TLS management across clusters and tenants.
---

# cert-manager Mastery (Senior → Principal)

## Operate

- Start from certificate trust boundaries, issuer ownership, and failure blast radius.
- Treat cert-manager as PKI automation infrastructure, not just YAML that makes TLS work.
- Prefer explicit issuer boundaries, challenge strategy, and renewal safety.
- Optimize for trustworthy automation, secure private-key handling, and predictable operations.

## Default Standards

- Issuer design must reflect trust and tenancy boundaries.
- Renewal and rotation behavior should be tested, not assumed.
- DNS and HTTP challenge strategy should match operational ownership.
- Secret and key handling require strong discipline.
- Shared certificate automation needs governance before scale.

## References

- Issuer architecture and trust boundaries: [references/issuer-architecture-and-trust-boundaries.md](references/issuer-architecture-and-trust-boundaries.md)
- ACME, DNS, and HTTP challenge strategy: [references/acme-dns-and-http-challenge-strategy.md](references/acme-dns-and-http-challenge-strategy.md)
- Certificate lifecycle, renewal, and rotation: [references/certificate-lifecycle-renewal-and-rotation.md](references/certificate-lifecycle-renewal-and-rotation.md)
- Secret handling and private-key security: [references/secret-handling-and-private-key-security.md](references/secret-handling-and-private-key-security.md)
- Multi-tenant governance and policy: [references/multi-tenant-governance-and-policy.md](references/multi-tenant-governance-and-policy.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
