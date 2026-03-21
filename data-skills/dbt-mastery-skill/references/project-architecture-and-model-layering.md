# Project Architecture and Model Layering (dbt)

## Rules

- Project structure should reflect domain ownership, semantic clarity, and consumer expectations.
- Staging, intermediate, marts, and domain layers should exist for reasons operators and analysts can explain.
- Avoid model sprawl with unclear purpose or ownership.
- dbt architecture should help teams reason about lineage and trust.

## Practical Guidance

- Define naming and directory conventions that map to business domains.
- Separate reusable patterns from local transformations thoughtfully.
- Keep consumer-facing models stable and well-documented.
- Review where one project should split or where multiple projects should consolidate.

## Principal Review Lens

- Which layer is carrying too much mixed intent today?
- Are we building coherent data products or just accumulating SQL files?
- What ownership boundary is least clear?
- Which refactor most improves trust and discoverability?
