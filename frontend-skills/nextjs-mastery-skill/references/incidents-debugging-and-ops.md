# Incidents, Debugging, and Production Operations in Next.js

## Principle

Frontend incidents in Next.js often involve multiple layers at once: rendering strategy, cache behavior, runtime environment, auth, and backend integration.

## Operational Questions to Answer Fast

During a real incident, operators should quickly determine:

- which route class is affected
- whether the issue is runtime-specific, cache-specific, or auth-specific
- whether the blast radius is public, authenticated, or role-specific
- whether rollback, feature degradation, or cache invalidation is the safest first move

## Common Incident Classes

- hydration or client-boundary regressions
- auth or middleware loops
- route handler failures
- stale content after mutation or deployment
- platform/runtime-specific regressions

## Triage Heuristics

### Separate rendering failures from data freshness failures

Many incidents look like "the page is broken" when the real distinction is:

- HTML/rendering shape is wrong
- client boundary behavior is wrong
- cache freshness or personalization is wrong
- backend integration returned unexpected data

### Correlate with the last meaningful architectural change

Pay special attention to:

- cache or revalidation changes
- new `use client` boundaries
- middleware or auth changes
- route handler changes
- runtime target changes

### Reduce blast radius before perfect certainty

Potential first moves include:

- disable non-critical interactive regions
- reduce cache aggressiveness
- bypass a broken route handler path
- simplify a redirect or auth branch temporarily

## Triage Questions

- is the issue route-specific or systemic?
- did runtime, cache, or auth behavior change recently?
- is the regression only visible in production traffic shape?
- can the route degrade safely while preserving core user journeys?

## Operator Guidance

- correlate incidents to route class, release version, and runtime target
- prefer reducing blast radius before chasing perfect root-cause detail
- keep rendering and cache assumptions visible in logs and dashboards where possible
- distinguish incidents caused by route architecture from incidents caused by backend or platform behavior

## Review Questions

- what would operators disable first if this route failed under real traffic?
- which route assumption is currently least observable in production?
- what incident class would be hardest to attribute correctly today?
