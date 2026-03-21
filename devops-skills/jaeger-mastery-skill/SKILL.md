---
name: jaeger-principal-engineer
description: |
  Principal/Senior-level Jaeger playbook for distributed tracing architecture, sampling, storage backends, operational workflows, and production trace platform governance.
  Use when: designing tracing systems, operating Jaeger deployments, reviewing sampling and storage tradeoffs, or debugging distributed trace reliability.
---

# Jaeger Mastery (Senior → Principal)

## Operate

- Start from the incident and debugging questions traces must answer.
- Treat collectors, agents, query path, storage backend, and sampling as a unified system.
- Prefer operational clarity over trace feature sprawl.
- Design for trustworthy traces during outages, not only normal days.

## Default Standards

- Sampling must reflect business-critical workflows.
- Storage and retention must match actual debugging value.
- Multi-tenant and access boundaries should be explicit where shared.
- Query UX should align with real investigation paths.
- Missing or partial traces must be diagnosable.

## References

- Trace architecture and component roles: [references/trace-architecture-and-component-roles.md](references/trace-architecture-and-component-roles.md)
- Sampling strategy and critical workflows: [references/sampling-strategy-and-critical-workflows.md](references/sampling-strategy-and-critical-workflows.md)
- Storage backends and retention tradeoffs: [references/storage-backends-and-retention-tradeoffs.md](references/storage-backends-and-retention-tradeoffs.md)
- Query workflows and trace usability: [references/query-workflows-and-trace-usability.md](references/query-workflows-and-trace-usability.md)
- Multi-tenant governance and access control: [references/multi-tenant-governance-and-access-control.md](references/multi-tenant-governance-and-access-control.md)
- Correlation and platform integration: [references/correlation-and-platform-integration.md](references/correlation-and-platform-integration.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
