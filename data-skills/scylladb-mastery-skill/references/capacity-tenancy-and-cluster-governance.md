# Capacity, Tenancy, and Cluster Governance

## Rules

- Capacity planning must include node loss, maintenance, skew, and growth by tenant/workload class.
- Shared clusters need governance over high-volume and high-hotspot workloads.
- One tenant should not silently dominate shard or node behavior.
- Governance should improve predictability and supportability.

## Practical Guidance

- Track hot tenants, hot partitions, and storage-heavy tables explicitly.
- Forecast growth using more than average rates.
- Align tenant boundaries with blast radius and support realities.
- Review new workloads for model and maintenance fit before onboarding.

## Principal Review Lens

- Which tenant can create the biggest cluster incident today?
- Are capacity models hiding skew behind averages?
- What governance gap most threatens cluster stability?
- What workload should be isolated or redesigned first?
