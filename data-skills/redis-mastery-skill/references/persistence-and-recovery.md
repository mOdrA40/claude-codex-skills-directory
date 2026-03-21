# Persistence and Recovery (Redis)

## Rules

- Pick RDB, AOF, or both based on recovery objectives.
- Persistence settings are business decisions, not defaults.
- Test restart behavior and data loss windows.
- Do not promise durability beyond configured reality.

## Principal Review Lens

- What data loss is possible on crash right now?
- How long is restart and warm-up under realistic memory size?
- Are teams using Redis as durable storage by accident?
