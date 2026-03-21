# Consistency, Topology, and Multi-DC Design

## Rules

- Consistency level choices should reflect true business semantics.
- Multi-DC design increases resilience and operational complexity together.
- Topology should remain understandable to operators and application teams.
- Cross-DC behavior must be tested with real clients.

## Practical Guidance

- Match local versus global correctness needs deliberately.
- Review network and latency realities before cross-DC rollout.
- Keep RF, placement, and failure assumptions explicit.
- Document degraded-mode behavior under DC or node impairment.

## Topology Heuristics

### Multi-DC is an operational commitment

Using multiple data centers changes latency posture, failure semantics, runbook burden, and what application teams must understand about partial impairment.

### Consistency levels define user truth under stress

The meaningful question is not just what consistency level is configured, but what users and downstream systems may safely believe when one region or path is degraded.

### Explainability matters in low-latency systems too

High-performance systems become dangerous when teams optimize for speed while leaving topology and failure behavior too opaque for operators and service owners.

## Common Failure Modes

### Multi-DC by aspiration

The architecture adopts cross-DC complexity before the organization has the operational maturity or real product need to support it safely.

### Consistency assumption drift

Applications evolve stronger correctness expectations than the original ScyllaDB deployment model was designed to uphold.

### Regional impairment under-modeled

Teams think in terms of healthy vs down, while the actual danger comes from slow, unstable, or asymmetric cross-DC behavior.

## Principal Review Lens

- What promise does the current consistency posture really make?
- Which topology assumption is least tested today?
- Are we using multi-DC for real need or speculative architecture theater?
- Can the team explain behavior during partial regional failure?
- Which application assumption would fail first if one DC became slow instead of fully unavailable?
