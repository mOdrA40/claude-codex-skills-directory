---
name: nginx-ingress-principal-engineer
description: |
  Principal/Senior-level NGINX Ingress playbook for traffic management, edge reliability, multi-tenant ingress governance, security controls, performance tuning, and production Kubernetes ingress operations.
  Use when: designing ingress topology, debugging traffic routing, tuning edge behavior, securing external access, or operating shared ingress platforms.
---

# NGINX Ingress Mastery (Senior → Principal)

## Operate

- Start from request flow, trust boundaries, tenant separation, and failure blast radius.
- Treat ingress as product-facing critical infrastructure.
- Prefer explicit routing, sane defaults, and clear ownership of TLS, auth, and rewrite behavior.
- Optimize for safe change rollout, debuggability, and predictable traffic behavior.

## Default Standards

- Routing rules should be understandable by humans.
- Edge timeouts, retries, body limits, and buffering should match workload reality.
- Multi-tenant ingress needs policy before scale.
- Observability at the edge is mandatory.
- Controller upgrades and config changes should be reversible.

## References

- Ingress topology and ownership: [references/ingress-topology-and-ownership.md](references/ingress-topology-and-ownership.md)
- Routing, rewrites, and path behavior: [references/routing-rewrites-and-path-behavior.md](references/routing-rewrites-and-path-behavior.md)
- TLS, auth, and edge security: [references/tls-auth-and-edge-security.md](references/tls-auth-and-edge-security.md)
- Performance tuning and traffic shaping: [references/performance-tuning-and-traffic-shaping.md](references/performance-tuning-and-traffic-shaping.md)
- Multi-tenant policy and platform governance: [references/multi-tenant-policy-and-platform-governance.md](references/multi-tenant-policy-and-platform-governance.md)
- Observability and debugging: [references/observability-and-debugging.md](references/observability-and-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
