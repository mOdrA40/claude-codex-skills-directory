# Query DSL, Filters, and Scoring

## Rules

- Distinguish filter context from scoring context clearly.
- Query logic should be explainable and measurable.
- Expensive queries need ownership and guardrails.
- Relevance tuning should not break operator trust in search latency.

## Common Failure Modes

- Scoring clauses used where simple filters would be better.
- Query templates accreting complexity nobody can explain.
- Nested or wildcard-heavy queries becoming cluster pain.
- Business ranking logic mixed with textual relevance without measurement.

## Principal Review Lens

- Which part of the query is expensive and which part actually improves results?
- Are we optimizing relevance with data or superstition?
- What query pattern will hurt the cluster worst under peak traffic?
- Can the team debug scoring under incident pressure?
