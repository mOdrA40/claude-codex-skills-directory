# Incident Runbooks (dbt)

## Cover at Minimum

- Critical model failure.
- Freshness or source outage.
- Bad package or macro rollout.
- Full-refresh / cost explosion.
- Contract break affecting downstream consumers.
- High-blast-radius model change.

## Incident Heuristics

### Separate execution failure from trust failure

Some dbt incidents stop runs outright. Others produce data that still materializes but should not be trusted by consumers.

### Protect critical datasets before broad rebuilds

The right first move is often to isolate or roll back high-blast-radius changes rather than trigger wide rebuilds that increase cost and confusion.

### Recovery must include consumer communication

Analytics trust does not recover automatically when a run turns green again. Consumers need clarity on freshness, semantics, and whether previously published data remains safe.

## Response Rules

- Restore trust in critical datasets before broad cleanup.
- Prefer targeted rollback or isolation over platform-wide panic changes.
- Preserve run logs, compiled SQL, and lineage evidence for RCA.
- Communicate clearly about data trust, freshness, and consumer impact.

## Common Failure Modes

### Green-run illusion

The job succeeds again, but contract confusion, stale upstreams, or business-semantics damage still make the output unsafe.

### Macro blast radius under-modeled

One package or macro change affects far more models than teams expected, and rollback logic is weaker than rollout logic.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider trust damage?
- What proves the analytics platform is healthy again?
- Are runbooks aligned with real transformation failure patterns?
- Which dbt incident still lacks an explicit consumer-trust recovery path?
