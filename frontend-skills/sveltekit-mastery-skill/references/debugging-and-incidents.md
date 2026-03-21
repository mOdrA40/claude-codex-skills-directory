# Debugging and Incidents in SvelteKit

## Principle

Production debugging in SvelteKit often spans route loading, action behavior, rendering strategy, and platform adapter differences.

## Incident Classes

- form action regressions
- redirect/auth flow confusion
- route-specific stale or missing data
- SSR/client mismatch assumptions
- deployment-adapter-specific failures

## Review Questions

- is the issue tied to route load, action flow, or platform runtime?
- did enhancement or adapter behavior change recently?
- how can blast radius be reduced while preserving core journeys?
