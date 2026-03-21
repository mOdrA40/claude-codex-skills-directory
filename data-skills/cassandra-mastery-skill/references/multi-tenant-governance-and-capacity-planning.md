# Multi-Tenant Governance and Capacity Planning

## Rules

- Shared Cassandra clusters need tenant-aware modeling, quotas, and ownership boundaries.
- Capacity planning must include repair, compaction, node loss, and topology growth.
- One tenant should not silently dominate partitions or operational cost.
- Governance should focus on predictable load and supportability.

## Practical Guidance

- Track hot tenants, large partitions, and uneven write patterns.
- Forecast growth by tenant and workload class.
- Align environment and tenant boundaries with blast radius reality.
- Standardize review of new high-volume tables.

## Principal Review Lens

- Which tenant can create the biggest cluster incident today?
- Are capacity models based on averages that hide hotspots?
- What governance gap most threatens cluster stability?
- What workload should be isolated or redesigned first?
