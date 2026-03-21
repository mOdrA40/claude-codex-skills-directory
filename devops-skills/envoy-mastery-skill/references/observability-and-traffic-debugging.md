# Observability and Traffic Debugging (Envoy)

## Rules

- Proxy observability should expose request flow, route choice, upstream behavior, and failure attribution.
- Distinguish proxy symptoms from upstream service symptoms quickly.
- Access logs, stats, and traces should support real incident debugging.
- Avoid metrics overload that obscures the few signals operators need most.

## Useful Signals

- Response codes, upstream reset reasons, retries, latency buckets, circuit-breaker status, connection state, and config version.
- Correlate proxy signals with service and platform telemetry.
- Standardize common dashboards for edge and internal proxy roles.
- Keep debug workflows documented for live traffic issues.

## Principal Review Lens

- Can on-call prove whether the proxy or upstream is at fault quickly?
- Which missing signal causes most traffic debugging delay?
- Are access logs tuned for forensic value or just verbosity?
- What observability improvement most reduces MTTR?
