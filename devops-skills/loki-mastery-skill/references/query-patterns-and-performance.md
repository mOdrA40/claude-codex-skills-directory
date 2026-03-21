# Query Patterns and Performance

## Rules

- Queries should be built around labels first, content search second.
- Expensive regex or broad scans should be deliberate and rare.
- Saved searches and dashboards should reflect common investigation workflows.
- Query performance matters because humans need answers during incidents.

## Common Mistakes

- Treating Loki like a full-text search engine with no architecture consequences.
- Broad queries across long ranges becoming platform-wide tax.
- Dashboards embedding expensive log queries everywhere.
- Teams blaming Loki for weak label strategy.

## Principal Review Lens

- Which query pattern causes the most hidden cost?
- Are users trained to query effectively or left to brute-force search?
- What dashboard query should be redesigned or precomputed?
- Can on-call get useful answers quickly under pressure?
