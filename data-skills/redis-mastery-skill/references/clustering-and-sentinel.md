# Clustering and Sentinel (Redis)

## Rules

- HA topology should match failure tolerance and client behavior.
- Client libraries must be tested against failover and slot movement.
- Understand the operational cost of cluster versus single-node simplicity.
- Failover does not remove the need for degraded-mode planning.

## Principal Review Lens

- What client-visible errors appear during failover?
- Is topology complexity justified by the workload?
- Which maintenance event creates the biggest user impact?
