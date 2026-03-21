# Incident Runbooks (NGINX Ingress)

## Cover at Minimum

- TLS or certificate expiration issues.
- Bad routing rollout.
- Controller crash or degraded reload behavior.
- Upstream timeout floods.
- DNS or external load balancer mismatch.
- Tenant-specific edge misconfiguration with shared platform impact.

## Response Rules

- Restore safe traffic flow before optimizing perfect config.
- Prefer known-good rollback paths and targeted isolation.
- Capture config, logs, and metrics before large emergency edits.
- Communicate clearly whether the problem is edge, DNS, or upstream service health.

## Principal Review Lens

- Can responders isolate one tenant or route without breaking everyone else?
- Which emergency action most risks widening customer impact?
- What evidence proves the edge is healthy after mitigation?
- Are runbooks practical for real traffic incidents, not only theory?
