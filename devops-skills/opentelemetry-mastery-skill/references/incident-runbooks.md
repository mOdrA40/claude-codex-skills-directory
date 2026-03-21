# Incident Runbooks (OpenTelemetry)

## Cover at Minimum

- Broken context propagation.
- Collector overload or crash loops.
- Exporter failures and telemetry drops.
- Bad sampling or transform rollouts.
- Trace correlation gaps during major incidents.
- Vendor/backend outages affecting telemetry pipelines.

## Response Rules

- Restore trustworthy critical-path telemetry before optimizing completeness.
- Prefer rollback of risky transforms over ad-hoc complex fixes.
- Preserve evidence about dropped, sampled, or malformed telemetry.
- Communicate clearly when traces are partially trustworthy rather than silently wrong.

## Principal Review Lens

- Can on-call prove whether telemetry is missing or merely not yet visible?
- Which action restores the most truth with the least extra risk?
- What signals confirm end-to-end health of the pipeline?
- Are runbooks written for real incidents or only for architecture diagrams?
