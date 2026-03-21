# Correlation with Metrics and Traces

## Rules

- Logs become more valuable when they connect clearly to metrics and traces.
- Correlation identifiers and service identity dimensions should be standardized.
- Do not assume all backends tell equally complete truth.
- Build workflows that let responders move from symptom to evidence quickly.

## Practical Guidance

- Preserve trace or request context where it materially improves debugging.
- Align service, environment, region, and tenant identity across signals.
- Link common dashboards and traces to representative log views.
- Make gaps explicit when sampling or retention limits correlation.

## Principal Review Lens

- Can on-call move from red metric to relevant logs fast?
- Which missing field most hurts cross-signal debugging?
- Are teams overlogging because correlation is weak?
- What correlation path is most brittle today?
