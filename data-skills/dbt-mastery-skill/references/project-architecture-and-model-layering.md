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

## Architecture Heuristics

### Layering should encode trust and ownership

Good dbt structure helps teams answer which models are raw, which are transforming domain semantics, which are consumer-facing, and who owns each layer.

### Stable marts need stronger contracts than transient layers

Not every model deserves the same compatibility promise. Consumer-facing layers should be far more stable, documented, and governed than transient transformation steps.

### Avoid turning dbt into a SQL junk drawer

Without strong layering discipline, projects slowly accumulate mixed-intent models that are hard to discover, test, refactor, and trust.

## Common Failure Modes

### Layer naming without semantic meaning

Directories exist, but they do not actually help teams understand business ownership or trust level.

### Consumer-facing instability hidden in intermediate layers

Models behave like public interfaces even though they are treated as internal and changed too casually.

### Domain boundaries blurred by convenience

One team keeps adding logic into shared areas until ownership and review discipline become weak.

## Principal Review Lens

- Which layer is carrying too much mixed intent today?
- Are we building coherent data products or just accumulating SQL files?
- What ownership boundary is least clear?
- Which refactor most improves trust and discoverability?
- Which model layer currently behaves like a platform boundary without being governed like one?
