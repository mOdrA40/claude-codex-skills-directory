# Observability and Debugging

## Rules

- Edge observability should reveal routing decisions, latency contribution, TLS behavior, and upstream outcomes.
- Logs and metrics must support tenant, host, path, and controller-level debugging where appropriate.
- Debugging ingress problems should distinguish DNS, client, TLS, controller, and upstream failure layers.
- Access logs should be useful without becoming a privacy or cost disaster.

## Useful Signals

- Request rate, latency percentiles, response class, upstream error rate, config reload status, certificate expiry, and saturation signals.
- Correlate ingress data with app traces and service metrics.
- Preserve enough context to investigate edge-specific incidents.
- Make common failure paths visible on standard dashboards.

## Principal Review Lens

- Can operators prove whether failure is at edge or upstream quickly?
- Which missing signal most slows ingress incidents today?
- Are logs trustworthy enough for forensic debugging?
- What observability improvement would most reduce MTTR at the edge?
