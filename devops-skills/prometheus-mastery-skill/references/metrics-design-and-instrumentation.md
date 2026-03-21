# Metrics Design and Instrumentation (Prometheus)

## Core Principles

- Instrument business-critical paths, saturation signals, error modes, and latency boundaries first.
- Prefer a small number of trustworthy metrics over a flood of semi-random counters.
- Labels should represent bounded dimensions with stable operational meaning.
- Histograms and summaries should be chosen based on aggregation needs, cost, and latency analysis goals.

## Non-Negotiable Rules

- Never attach high-cardinality identifiers such as raw user IDs, request IDs, or timestamps as labels.
- Metric names must remain semantically stable across releases.
- One metric should answer one class of question clearly.
- Exporters and application instrumentation should be reviewed like API design.

## Principal Review Lens

- Which product or incident decision becomes easier because this metric exists?
- Which label is most likely to explode cardinality in six months?
- Are we measuring user pain, infrastructure churn, or both without separation?
- If one metric disappeared, what blind spot would hurt the team most?
