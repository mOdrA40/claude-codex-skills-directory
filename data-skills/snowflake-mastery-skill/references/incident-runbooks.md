# Incident Runbooks (Snowflake)

## Cover at Minimum

- Runaway warehouse cost incident.
- Shared data product breakage.
- Access control or masking failure.
- Critical pipeline or query outage.
- Accidental privilege escalation.
- High-blast-radius sharing or schema change.

## Incident Heuristics

### Separate trust incidents from performance incidents

Snowflake incidents are not all about query failure. Some are fundamentally about trust, access, masking, sharing boundaries, or finance exposure.

### Protect governed access before convenience

The fastest recovery action is not always the safest if it weakens masking, privilege boundaries, or trust in critical data products.

### Recovery must include cost containment

A platform is not recovered if critical queries run again but runaway warehouse economics or high-blast-radius access posture remain unresolved.

## Response Rules

- Stabilize business-critical access and workloads first.
- Prefer targeted isolation over broad platform panic.
- Preserve evidence around cost spikes, access changes, and failed queries.
- Communicate clearly about data availability, access, and trust impact.

## Common Failure Modes

### Performance-first response to governance incident

The team restores access or throughput quickly but leaves trust, compliance, or data product correctness ambiguous.

### Cost spike normalized as temporary noise

The immediate emergency passes, but the workload isolation or warehouse policy flaw that caused the spike remains.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks trust or compliance damage?
- What proves the platform is healthy again?
- Are runbooks aligned with real platform failure modes?
- Which Snowflake incident still lacks an explicit safe-first governance response?
