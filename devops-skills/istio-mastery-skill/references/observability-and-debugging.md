# Observability and Debugging (Istio)

## Rules

- Mesh observability should clarify request paths, policy decisions, and failure layers.
- Distinguish application issues from proxy, policy, DNS, or certificate issues quickly.
- Standard dashboards should reveal dependency errors, latency tax, and config rollout effects.
- Proxy metrics should support operations without drowning humans in noise.

## Useful Signals

- Request success rates, latency, retry counts, reset reasons, mTLS failures, config push status, and control plane health.
- Correlate mesh signals with app metrics and traces.
- Make common rollout and policy errors obvious to operators.
- Keep debugging workflows documented for sidecar and control-plane boundaries.

## Principal Review Lens

- Can on-call prove where failure sits in the request path quickly?
- Which missing signal causes most mesh confusion today?
- Are teams over-attributing problems to Istio because app signals are weak?
- What debug workflow most needs simplification?
