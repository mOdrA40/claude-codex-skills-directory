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

## Principal Review Lens

- What promise does the current consistency posture really make?
- Which topology assumption is least tested today?
- Are we using multi-DC for real need or speculative architecture theater?
- Can the team explain behavior during partial regional failure?
