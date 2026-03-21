# Offline Sync and Persistence in Hybrid Apps

## Principle

Hybrid apps often face the same sync problems as native apps, but with extra complexity from browser storage models, plugin variability, and mobile lifecycle behavior.

## Common Failure Modes

- assuming web persistence patterns behave identically under native lifecycle pressure
- local data surviving longer than the product model expects
- background/resume behavior causing confusing sync states

## Review Questions

- what persists across restarts and upgrades?
- what is the source of truth for critical workflows?
- what happens when sync resumes after long offline periods?
