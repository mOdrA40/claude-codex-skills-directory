# Services and Ingress

## Rules

- Service topology should be simple enough to reason about under incident pressure.
- Ingress behavior must align with security and traffic-routing needs.
- Path and host routing need explicit ownership.
- Timeouts, retries, and sticky behavior are product decisions too.

## Principal Review Lens

- Which routing layer is most likely to hide the real failure?
- Are ingress defaults accidentally unsafe?
- Can on-call trace request flow end-to-end quickly?
