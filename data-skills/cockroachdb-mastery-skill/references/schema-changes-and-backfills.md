# Schema Changes and Backfills (CockroachDB)

## Rules

- Treat backfill cost as production work, not metadata trivia.
- Roll out schema changes with realistic traffic in mind.
- Validate index and column changes against latency-sensitive paths.
- Prefer safe rollout choreography over one-shot migration bravado.

## Principal Review Lens

- What user-visible latency tax appears during backfill?
- Can the rollout pause or recover safely?
- Is the schema change operationally boring under peak load?
