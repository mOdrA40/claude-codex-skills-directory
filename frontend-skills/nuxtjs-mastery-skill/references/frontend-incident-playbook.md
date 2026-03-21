# Frontend Incident Playbook for Nuxt Applications

## Principle

Frontend incidents are not only crashes. They include hydration mismatch spikes, cache poisoning, broken route guards, auth loops, and degraded interactive latency.

## First Questions

- is the issue SSR, hydration, or client-only navigation?
- did a config or route-rule change ship recently?
- is one route class or one browser family most affected?
- is stale cached data part of the failure?

## Common Incident Classes

- hydration mismatch
- infinite redirect/auth loops
- query/cache invalidation regressions
- oversized bundle or route chunk regressions
- browser-only code running too early

## Triage Heuristics

### First isolate the blast radius

Determine whether the issue affects:

- one route class
- one browser family
- one authentication state
- one deployment region or cache path

### Correlate with the last real change

Look for:

- route-rule changes
- cache behavior changes
- auth middleware changes
- SSR/client boundary changes
- bundle growth or dependency shifts

### Prefer fast reduction of user pain

If possible:

- disable a broken enhancement path
- reduce redirect complexity
- fall back to simpler rendering behavior
- bypass a non-critical data dependency

## Review Questions

- what is the fastest way to reduce blast radius?
- is the incident caused by SSR, hydration, routing, or cached state behavior?
- what recent change most likely altered route behavior or first render assumptions?
