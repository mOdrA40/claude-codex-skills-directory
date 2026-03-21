# Reliability and Operations (Istio)

## Operational Defaults

- Monitor control plane health, config push status, proxy connectivity, certificate rotation, and traffic error signals.
- Stage control plane upgrades carefully and verify dataplane compatibility.
- Separate control plane incidents from single-tenant or single-policy incidents quickly.
- Document safe rollback and emergency disable patterns before they are needed.

## Run-the-System Thinking

- Mesh platforms deserve SLOs if many teams depend on them.
- Upgrades and policy rollouts should be reversible and observable.
- Capacity planning must include control plane and dataplane tax.
- On-call should know which mesh features are safe to disable during crisis.

## Principal Review Lens

- Which mesh failure mode has the highest blast radius?
- Can the team upgrade or recover control plane safely?
- What operational habit would most improve trust in the mesh?
- Are we operating a platform or accumulating traffic magic debt?
