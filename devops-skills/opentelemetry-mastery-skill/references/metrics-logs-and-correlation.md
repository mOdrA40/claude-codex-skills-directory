# Metrics, Logs, and Correlation

## Rules

- OpenTelemetry should support correlation across signals, not create disconnected silos.
- Resource attributes and identity dimensions must line up with how teams debug systems.
- Logs should carry enough context to connect with traces where appropriate.
- Metrics generated from telemetry pipelines must remain trustworthy and cost-aware.

## Design Heuristics

- Standardize service.name, environment, region, version, and deployment identity.
- Preserve trace IDs in logs where it materially helps investigation.
- Avoid duplicating every signal into every backend without a reason.
- Make correlation pathways obvious in dashboards and runbooks.

## Principal Review Lens

- Can an operator move from a red metric to the right traces and logs quickly?
- Which correlation link is weakest today?
- Are teams over-instrumenting because correlation standards are unclear?
- What cross-signal gap most hurts incident response?
