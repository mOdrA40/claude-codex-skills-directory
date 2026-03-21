# Loaders, Actions, and Data Flow in Remix

## Principle

Remix becomes elegant when loaders and actions have clear ownership. Without that, route data and mutation logic turn into hidden coupling.

## Common Failure Modes

- loaders fetching too broadly
- actions that perform vague orchestration across unrelated concerns
- unclear data freshness after mutation or navigation

## Ownership Heuristics

### Loader ownership

Loaders should own the data needed to render their route boundary, not every nearby concern that might be useful eventually.

### Action ownership

Actions should represent explicit mutation boundaries with clear validation, side effects, and recovery behavior.

### Mutation aftermath

Teams should define:

- which UI surfaces refresh
- which route data becomes stale
- whether navigation is part of the success path
- what degraded behavior exists on failure

## Review Questions

- what data belongs to this route boundary?
- what action result should refresh which UI surface?
- who owns failure mapping?
- what route or action currently hides too much orchestration behind a simple interface?
