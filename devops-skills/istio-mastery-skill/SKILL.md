---
name: istio-principal-engineer
description: |
  Principal/Senior-level Istio playbook for service mesh architecture, traffic policy, identity, security, observability, and operating multi-tenant mesh platforms in production.
  Use when: designing mesh adoption, reviewing traffic policy and mTLS posture, debugging sidecar or ambient behavior, or operating Istio at scale.
---

# Istio Mastery (Senior → Principal)

## Operate

- Start from why the mesh exists: security boundaries, traffic control, platform consistency, or observability enrichment.
- Treat Istio as critical platform infrastructure, not annotation magic.
- Prefer explicit traffic and security policy with clear ownership.
- Optimize for debuggability, rollout safety, and blast-radius control.

## Default Standards

- Mesh adoption should be intentional, not blanket by habit.
- mTLS and identity policy must reflect trust boundaries.
- Traffic rules should remain understandable under incident pressure.
- Sidecar or ambient mode operational cost must be explicit.
- Mesh observability should help incident response, not bury teams in proxy internals.

## References

- Mesh architecture and adoption boundaries: [references/mesh-architecture-and-adoption-boundaries.md](references/mesh-architecture-and-adoption-boundaries.md)
- Traffic management and rollout control: [references/traffic-management-and-rollout-control.md](references/traffic-management-and-rollout-control.md)
- Security, mTLS, and identity: [references/security-mtls-and-identity.md](references/security-mtls-and-identity.md)
- Sidecars, ambient mode, and dataplane tradeoffs: [references/sidecars-ambient-mode-and-dataplane-tradeoffs.md](references/sidecars-ambient-mode-and-dataplane-tradeoffs.md)
- Multi-tenant platform governance: [references/multi-tenant-platform-governance.md](references/multi-tenant-platform-governance.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
