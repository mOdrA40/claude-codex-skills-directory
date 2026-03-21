# Incident Runbooks (Envoy)

## Cover at Minimum

- Bad route or cluster rollout.
- TLS handshake or certificate incident.
- Retry storm or overload misbehavior.
- xDS/config delivery failure.
- Proxy crash or degraded fleet behavior.
- Upstream attribution confusion during major outage.

## Response Rules

- Stabilize critical traffic paths before config cleanup.
- Prefer known-good rollback paths over speculative tuning.
- Preserve config versions, logs, and request-failure evidence.
- Communicate clearly whether failure is proxy, config plane, or upstream service health.

## Principal Review Lens

- Can responders isolate the problem layer quickly?
- Which emergency action risks compounding traffic failure?
- What evidence proves the proxy fleet is healthy again?
- Are runbooks strong enough for platform-wide incidents?
