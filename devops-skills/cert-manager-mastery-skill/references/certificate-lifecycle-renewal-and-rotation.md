# Certificate Lifecycle, Renewal, and Rotation

## Rules

- Renewal should be boring and observable long before expiration windows become dangerous.
- Rotation behavior must be compatible with workload reload and deployment patterns.
- Certificate lifetime, overlap, and revocation assumptions should be explicit.
- Teams should know what happens when renewal fails or is delayed.

## Practical Guidance

- Test end-to-end renewal and secret update behavior.
- Track certificates by criticality, expiry horizon, and owner.
- Align workload rollout behavior with key/cert replacement patterns.
- Keep incident paths for expiring or failed certs rehearsed.

## Principal Review Lens

- Which critical cert is most likely to expire unexpectedly?
- Are workloads actually safe during rotation?
- What renewal failure path is least visible today?
- Which lifecycle control most improves trust in automation?
