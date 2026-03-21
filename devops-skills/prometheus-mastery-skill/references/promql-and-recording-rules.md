# PromQL and Recording Rules

## Query Design Rules

- PromQL should be readable by operators under pressure.
- Encode SLOs, burn rates, saturation, and capacity analysis in reusable queries.
- Use recording rules for expensive or repeatedly used logic, not to hide poor query discipline.
- Distinguish instant troubleshooting queries from long-term reporting queries.

## Failure Modes

- Overly complex expressions become impossible to debug during incidents.
- Bad rate windows or aggregation order can silently distort conclusions.
- Recording rules can fossilize wrong assumptions if not reviewed.
- Query cost can become platform pain at scale.

## Principal Review Lens

- Can another team explain this query in under two minutes?
- Is the aggregation order mathematically correct for the question being asked?
- Are we turning dashboard convenience into hidden system logic?
- Which expensive query should become a rule before it becomes a cluster tax?
