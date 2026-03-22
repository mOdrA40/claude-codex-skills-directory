# Zero-Downtime Schema and Contract Migrations in Node.js Services

## Purpose

Schema and contract changes are a common place where code that looks correct becomes unsafe in production. Node.js services must assume mixed-version traffic during rollouts.

## Core Principle

Prefer expand-and-contract migrations.

That means:

- add compatible schema first
- deploy code that can work with both old and new shapes
- backfill safely if needed
- remove old shape only after compatibility window closes

## Rules

- avoid releases that require every instance to switch at once
- keep old and new event readers compatible where possible
- separate schema rollout from code cleanup
- define rollback posture before executing risky migration
- treat queue events and webhooks as compatibility surfaces too

## Agent Checklist

Before approving or generating migration-related code, ask:

- can old and new versions coexist?
- can rollback happen without schema surgery?
- will background workers process both versions safely?
- are dual-read or dual-write patterns required?
- where is the compatibility window documented?

## Anti-Pattern

```text
❌ BAD
Deploy code that expects a new non-null column before all writers and backfills are safe.

✅ GOOD
Add nullable/backward-compatible shape first, deploy tolerant code, backfill, enforce later.
```
