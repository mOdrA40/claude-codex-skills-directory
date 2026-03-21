# Correlation and Platform Integration

## Rules

- Traces gain most value when integrated into broader observability workflows.
- Metrics, logs, and trace entry points should share stable identity dimensions.
- Platform integration should shorten MTTR, not fragment investigation.
- Correlation design should acknowledge where trace gaps exist.

## Practical Guidance

- Align service identity, region, environment, and deployment metadata.
- Integrate with dashboards and logs where it matters most.
- Keep correlation workflows simple and repeatable.
- Make the limits of trace completeness visible to users.

## Principal Review Lens

- Can on-call pivot from symptom to trace fast enough?
- Which integration point is weakest today?
- Are teams overtrusting traces because the UI looks polished?
- What platform integration most improves investigation quality?
