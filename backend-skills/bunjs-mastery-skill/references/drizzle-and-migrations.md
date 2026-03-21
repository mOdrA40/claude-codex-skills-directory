# Drizzle and Migration Safety

## Principle

ORM convenience should never hide schema risk. Production backend systems need explicit migration discipline, compatibility rules, and rollback awareness.

## Rules

- treat schema changes as deploy events, not code-only changes
- prefer additive changes first
- avoid destructive changes in the same deploy as new code paths
- make read/write compatibility explicit during rollout
- verify background workers and API handlers remain schema-compatible

## Bad vs Good

```text
❌ BAD
Rename or drop a column in one step and deploy app + migration together.

✅ GOOD
Use expand-and-contract with compatibility windows.
```

## Review Questions

- Can old and new app versions coexist safely?
- What is the rollback path if the new code fails after migration?
- Are queued jobs serialized with old schema assumptions?
