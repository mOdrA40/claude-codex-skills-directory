# Incident Runbooks (dbt)

## Cover at Minimum

- Critical model failure.
- Freshness or source outage.
- Bad package or macro rollout.
- Full-refresh / cost explosion.
- Contract break affecting downstream consumers.
- High-blast-radius model change.

## Response Rules

- Restore trust in critical datasets before broad cleanup.
- Prefer targeted rollback or isolation over platform-wide panic changes.
- Preserve run logs, compiled SQL, and lineage evidence for RCA.
- Communicate clearly about data trust, freshness, and consumer impact.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks wider trust damage?
- What proves the analytics platform is healthy again?
- Are runbooks aligned with real transformation failure patterns?
