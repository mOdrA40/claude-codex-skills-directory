# Query DSL and Scoring (Elasticsearch)

## Rules

- Distinguish filter context from relevance scoring clearly.
- Scoring logic should be explainable to humans.
- Expensive queries need guardrails and ownership.
- Benchmark real query sets, not demo searches.

## Relevance Heuristics

### Retrieval and ranking are different jobs

Filters should narrow the candidate set efficiently. Scoring should then express why one matching document is better than another. Mixing those responsibilities carelessly creates both cost and confusion.

### Explainability is an operational requirement

If engineers or operators cannot explain why a clause exists and what it contributes, the scoring model is already becoming too opaque for safe production tuning.

### Tune with real query mixes

Search behavior should be evaluated against actual user queries, traffic distribution, and business ranking goals, not idealized demos.

## Common Failure Modes

### Boost pile-up

The team keeps adding boosts and should-clauses until relevance gets harder to reason about and latency grows without clear benefit.

### Filter cost hidden inside scoring complexity

Queries ask the engine to do expensive ranking work before enough narrowing happens.

### Incident-debugging opacity

The search system degrades, but nobody can quickly tell which clause or ranking idea is responsible.

## Principal Review Lens

- Which clause dominates cost and which dominates relevance?
- Are we filtering efficiently before scoring?
- Is scoring logic understandable enough to debug under pressure?
- What clause would we remove first if search latency became unacceptable tomorrow?
