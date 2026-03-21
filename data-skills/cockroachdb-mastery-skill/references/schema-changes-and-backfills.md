# Schema Changes and Backfills (CockroachDB)

## Rules

- Treat backfill cost as production work, not metadata trivia.
- Roll out schema changes with realistic traffic in mind.
- Validate index and column changes against latency-sensitive paths.
- Prefer safe rollout choreography over one-shot migration bravado.

## Rollout Heuristics

### Backfills are workload events

In CockroachDB, schema changes and backfills can compete with foreground traffic, amplify latency, and expose contention or locality assumptions that seemed harmless before rollout.

### Stage changes so recovery remains possible

Good choreography usually separates:

- schema introduction
- application compatibility rollout
- backfill observation period
- cleanup or constraint hardening

This reduces the blast radius of one mistaken assumption.

### Validate against hot paths, not only migration success

A migration is not safe just because it completes. It must remain boring under realistic write paths, retries, regional latency, and peak traffic.

## Common Failure Modes

### Metadata-only optimism

Teams treat the change as a lightweight schema operation while the real cost appears in backfill work, foreground contention, or index build side effects.

### Rollout without pause points

The change sequence has no clean observation boundary, so by the time trouble is visible the team has already coupled application and data changes too tightly.

### Background work hiding user pain

The backfill is technically progressing, but user-facing latency or retry pressure is already increasing on critical paths.

## Principal Review Lens

- What user-visible latency tax appears during backfill?
- Can the rollout pause or recover safely?
- Is the schema change operationally boring under peak load?
- What part of the rollout should be decoupled before this change is considered safe?
