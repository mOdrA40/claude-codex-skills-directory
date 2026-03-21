# Governance, Multi-Team DAG Ecosystems, and Ownership

## Rules

- Shared Airflow platforms need ownership, naming, and dependency discipline.
- One team should not create scheduler or queue chaos for everyone else.
- Governance should focus on supportability, not only restricting creativity.
- Ownerless DAGs are operational liabilities.

## Practical Guidance

- Track critical DAG owners and stale workflows.
- Standardize coding, packaging, and dependency patterns where useful.
- Limit high-risk dynamic DAG or plugin behavior.
- Make exception workflows explicit and reviewable.

## Principal Review Lens

- Which team or DAG has the highest platform blast radius?
- Are standards strong enough to prevent repeated anti-patterns?
- What ownerless DAG most threatens reliability today?
- Which governance change creates the most leverage?
