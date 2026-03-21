# Panels, Queries, and Performance

## Rules

- Panels should have a clear reason to exist and a comprehensible query behind them.
- Expensive dashboard queries become platform cost and user frustration.
- Visualization choice must match the question: trends, distributions, comparisons, or state.
- Variables and repetition should not create a hidden query explosion.

## Failure Modes

- One dashboard causing heavy load across Prometheus, logs, or traces backends.
- Broken units, misleading legends, and mathematically wrong aggregations.
- Panel duplication with inconsistent logic across teams.
- Overuse of templating that makes dashboards feel powerful but unreadable.

## Principal Review Lens

- Which panel is the most expensive relative to its value?
- Can a human verify the query logic quickly?
- Are we hiding poor observability design behind pretty visualizations?
- What dashboard would you trust least during a real outage and why?
