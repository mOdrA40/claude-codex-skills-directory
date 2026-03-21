# Incident Runbooks (Istio)

## Cover at Minimum

- Control plane degradation.
- Bad traffic policy rollout.
- mTLS or certificate failure.
- Sidecar injection or startup regression.
- Egress or ingress policy outage.
- Tenant-specific mesh config affecting shared platform behavior.

## Response Rules

- Stabilize critical request paths before broad policy cleanup.
- Prefer targeted rollback and isolation over emergency global edits.
- Preserve config state, control plane logs, and affected proxy signals.
- Communicate clearly whether failure is app, mesh policy, or control plane health.

## Principal Review Lens

- Can responders isolate blast radius within minutes?
- Which emergency action most risks making the outage wider?
- What evidence proves the mesh is healthy again?
- Are runbooks realistic for multi-team incidents?
