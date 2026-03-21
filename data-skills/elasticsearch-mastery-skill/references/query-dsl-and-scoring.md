# Query DSL and Scoring (Elasticsearch)

## Rules

- Distinguish filter context from relevance scoring clearly.
- Scoring logic should be explainable to humans.
- Expensive queries need guardrails and ownership.
- Benchmark real query sets, not demo searches.

## Principal Review Lens

- Which clause dominates cost and which dominates relevance?
- Are we filtering efficiently before scoring?
- Is scoring logic understandable enough to debug under pressure?
