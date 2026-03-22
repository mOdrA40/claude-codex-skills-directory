# Zero-Downtime Schema and Contract Migrations in Bun Services

## Core Principle

A Bun service should assume old and new versions coexist during rollout.

## Safe Sequence

1. add backward-compatible schema or event shape
2. deploy code tolerant of both old and new forms
3. backfill or dual-write only when justified
4. verify readers, workers, and webhooks remain compatible
5. remove legacy path only after safety window closes

## Agent Checklist

- can rollback happen cleanly?
- can workers and API handlers read both versions?
- are consumers deployed separately from request-serving instances?
- is the migration likely to create lock or backlog risk?
- is contract drift caught by validation or tests?

## Anti-Pattern

```text
❌ BAD
Deploy migration and application code assuming no mixed-version traffic exists.

✅ GOOD
Design rollout so old and new binaries, workers, and schemas can coexist safely.
```
