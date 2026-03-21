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

## Tuning Heuristics

### Keep expensive logic explainable

If the team cannot explain why one clause exists and what relevance value it adds, it is probably too expensive or too weakly governed.

### Separate retrieval from ranking intent

Filters, retrieval logic, and ranking/business boosts should remain conceptually distinct enough that operators can reason about latency and quality tradeoffs.

## Additional Failure Modes

### Relevance superstition

Teams keep adding boosts, should-clauses, and custom scoring behavior without controlled measurement, eventually making both search quality and latency harder to reason about.

### Incident-debugging blindness

The query works when traffic is light, but under pressure nobody can isolate which part of the template or scoring pipeline creates the cost.

## Principal Review Lens

- Which part of the query is expensive and which part actually improves results?
- Are we optimizing relevance with data or superstition?
- What query pattern will hurt the cluster worst under peak traffic?
- Can the team debug scoring under incident pressure?
- What clause would you remove first if latency became unacceptable tomorrow?
