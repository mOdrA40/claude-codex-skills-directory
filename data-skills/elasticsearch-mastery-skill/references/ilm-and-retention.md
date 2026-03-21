# ILM and Retention (Elasticsearch)

## Rules

- Retention is a product and compliance decision, not only a storage setting.
- ILM should align with data freshness, cost, and restore expectations.
- Hot/warm/cold tiering must justify its operational complexity.
- Deletion and rollover behavior should be predictable under load.

## Principal Review Lens

- Is retention longer than its real value?
- What tier transition creates most operational risk?
- Can the team explain ILM behavior during an incident?
