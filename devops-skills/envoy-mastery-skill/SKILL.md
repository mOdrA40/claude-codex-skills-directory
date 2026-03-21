---
name: envoy-principal-engineer
description: |
  Principal/Senior-level Envoy playbook for proxy architecture, listener/filter-chain design, resilience policy, edge/service-proxy operations, and production traffic debugging.
  Use when: designing L4/L7 proxy topologies, reviewing Envoy config, tuning traffic behavior, or operating Envoy-based platforms in production.
---

# Envoy Mastery (Senior → Principal)

## Operate

- Start from request flow, trust boundaries, and failure semantics.
- Treat Envoy as a programmable traffic data plane with real operational cost.
- Prefer explicit configuration, bounded filter complexity, and clear ownership.
- Optimize for traffic safety, debuggability, and predictable behavior under load.

## Default Standards

- Listener and route design should be explainable by humans.
- Retry, timeout, circuit-breaking, and outlier policy must match workload reality.
- Filter chains should stay as simple as possible.
- Config rollout and xDS behavior require operational discipline.
- Observability should reveal why requests were routed, failed, or delayed.

## References

- Listener architecture and filter chains: [references/listener-architecture-and-filter-chains.md](references/listener-architecture-and-filter-chains.md)
- Routing, clusters, and upstream policy: [references/routing-clusters-and-upstream-policy.md](references/routing-clusters-and-upstream-policy.md)
- Resilience controls and overload behavior: [references/resilience-controls-and-overload-behavior.md](references/resilience-controls-and-overload-behavior.md)
- Security, TLS, and identity at the proxy: [references/security-tls-and-identity-at-the-proxy.md](references/security-tls-and-identity-at-the-proxy.md)
- xDS, config delivery, and governance: [references/xds-config-delivery-and-governance.md](references/xds-config-delivery-and-governance.md)
- Observability and traffic debugging: [references/observability-and-traffic-debugging.md](references/observability-and-traffic-debugging.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
