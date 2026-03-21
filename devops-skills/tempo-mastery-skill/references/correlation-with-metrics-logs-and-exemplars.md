# Correlation with Metrics, Logs, and Exemplars

## Rules

- Trace platforms are most valuable when users arrive from a real symptom in metrics or logs.
- Exemplars and correlation identifiers must be designed deliberately.
- Identity dimensions across signals must align enough to support investigation.
- Correlation UX should shorten MTTR rather than add another navigation maze.

## Practical Guidance

- Preserve stable service, environment, and deployment identity.
- Use exemplars or links where they genuinely improve response workflows.
- Clarify where sampling weakens or breaks correlation.
- Standardize the most important trace-entry journeys for operators.

## Principal Review Lens

- Can operators move from a red graph to a useful trace in seconds?
- What correlation gap causes most investigation delay?
- Are exemplars configured where they matter most?
- What workflow most needs tighter integration with Grafana or logs?
