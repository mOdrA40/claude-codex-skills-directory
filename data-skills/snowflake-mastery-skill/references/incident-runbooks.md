# Incident Runbooks (Snowflake)

## Cover at Minimum

- Runaway warehouse cost incident.
- Shared data product breakage.
- Access control or masking failure.
- Critical pipeline or query outage.
- Accidental privilege escalation.
- High-blast-radius sharing or schema change.

## Response Rules

- Stabilize business-critical access and workloads first.
- Prefer targeted isolation over broad platform panic.
- Preserve evidence around cost spikes, access changes, and failed queries.
- Communicate clearly about data availability, access, and trust impact.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks trust or compliance damage?
- What proves the platform is healthy again?
- Are runbooks aligned with real platform failure modes?
