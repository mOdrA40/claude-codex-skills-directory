# Aggregation and Analytics (MongoDB)

## Rules

- Aggregation pipelines should be explicit about memory and sort cost.
- Analytical workloads can punish OLTP clusters if mixed carelessly.
- Precompute when runtime aggregation becomes operationally expensive.
- Keep business-critical analytics reproducible and testable.

## Principal Review Lens

- Is this aggregation competing with hot transactional traffic?
- Which stage forces spills or large intermediate results?
- Would materialization or ETL be safer here?
