# Reliability and Operations (NGINX Ingress)

## Operational Defaults

- Monitor controller health, reload behavior, request errors, latency, config churn, and certificate state.
- Keep controller upgrades staged and reversible.
- Separate platform-level edge incidents from single-service routing incidents quickly.
- Document ownership of controller config, ingress objects, and DNS dependencies.

## Run-the-System Thinking

- Edge platforms need SLOs if they front critical services.
- Config reload storms and annotation sprawl are real operational risks.
- Capacity planning must include TLS, request burstiness, and tenant skew.
- Emergency changes at the edge should be logged and reconciled.

## Principal Review Lens

- What edge failure mode has the highest customer blast radius?
- Which operational practice most improves ingress trustworthiness?
- Can the team upgrade controllers safely during normal business cadence?
- What hidden dependency makes ingress incidents harder than they should be?
