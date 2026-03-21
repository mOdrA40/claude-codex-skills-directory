# Provisioning and Dashboard as Code

## Rules

- Important dashboards, data sources, and alerting objects should be reproducible.
- Dashboard-as-code should improve reviewability, not bury intent in generated JSON.
- Provisioning pipelines need ownership and rollback posture.
- Manual edits in production should be rare, explainable, and reconcilable.

## Tradeoffs

- Fully generated dashboards can drift away from human readability.
- Manual creation is fast but often creates ownership and consistency debt.
- Review diffs should surface semantic change, not only formatting churn.
- Template systems should reduce repetition without obscuring critical logic.

## Principal Review Lens

- Can the team rebuild Grafana state confidently after disaster?
- What configuration is still tribal knowledge instead of code?
- Are dashboard reviews focused on operational meaning or merely syntax?
- Which manual workflow should be eliminated first?
