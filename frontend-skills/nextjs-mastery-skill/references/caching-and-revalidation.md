# Caching and Revalidation Strategy in Next.js

## Principle

Caching is one of the easiest ways to create performance wins and one of the easiest ways to create correctness bugs. Teams must be explicit about freshness, scope, and invalidation.

## Questions to Define Early

- which routes tolerate stale data
- which user-specific content must never leak across boundaries
- what triggers revalidation
- whether a mutation should update UI optimistically, revalidate, or both

## Freshness Heuristics

### Stable content routes

Good candidates for stronger caching when:

- content changes predictably
- personalization is limited
- editorial or product workflows can tolerate delayed freshness

### Interactive or mutation-heavy routes

These need tighter thinking about:

- mutation aftermath
- user trust in visible state
- stale data signaling
- revalidation blast radius

### User-specific surfaces

These require explicit guarantees so cached output does not create leakage or confusing personalization behavior.

## Decision Matrix

### Strong caching is usually safer when

- content changes infrequently
- users tolerate delayed freshness
- personalization is absent or tightly segmented
- rollback by cache invalidation is operationally simple

### Tight revalidation is usually safer when

- user trust depends on recent state
- mutations are common
- the route influences money, permissions, or workflow decisions
- stale content would create support or compliance risk

### Mixed strategy is justified when

- the route contains both stable and unstable regions
- layout or shell data can stay stable while feature panels refresh more aggressively
- the team can explain ownership of each freshness boundary clearly

## Common Failure Modes

### Cache works until auth enters the picture

A route performs well, but user-specific state and personalization make the cache model unsafe or confusing.

### Revalidation by superstition

The team triggers broad revalidation because exact ownership is unclear.

### Fast but untrustworthy UI

The product looks quick while serving outdated or inconsistent data after mutations.

### Revalidation fan-out

One mutation forces much broader freshness work than the product actually needs because cache ownership is vague.

### Invalidation asymmetry

One surface refreshes correctly while another surface stays stale, creating user confusion because the product appears internally inconsistent.

## Incident Heuristics

### Stale after mutation

Ask:

- was the wrong thing cached?
- was the right thing cached but not revalidated?
- was UI optimism masking backend truth?
- did auth or personalization change cache safety assumptions?

### Fast route, high support burden

Sometimes the route performs well technically but causes trust failures because users repeatedly see outdated or contradictory data.

## Bad vs Good

```text
❌ BAD
Revalidate broad route trees after every mutation because it is easier than modeling ownership.

✅ GOOD
Define freshness ownership by route segment, user impact, and mutation scope.
```

## Review Questions

- what freshness promise is each route making?
- what is the blast radius of an incorrect cache assumption?
- can operators explain why stale content appeared after a release?
- what cache behavior is currently fast but hardest to trust?
- which route segment has the worst tradeoff between speed and correctness right now?
