# Incident Runbooks (OPA Gatekeeper)

## Cover at Minimum

- Admission denial surge.
- Bad template or constraint rollout.
- Webhook or controller degradation.
- Audit drift explosion.
- Exception management failure.
- Cluster-critical deployment blocked by policy.

## Response Rules

- Restore safe cluster change capability before policy cleanup elegance.
- Prefer targeted rollback or exemption over broad panic disabling.
- Preserve deny evidence, template versions, and cluster-change context.
- Communicate clearly whether failure is policy, platform health, or workload config.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks weakening important controls too far?
- What proves the policy platform is healthy again?
- Are runbooks realistic for production admission incidents?
