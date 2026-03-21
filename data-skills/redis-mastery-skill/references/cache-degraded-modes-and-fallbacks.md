# Cache Degraded Modes and Fallbacks (Redis)

## Principle

A cache is only production-ready when the system can explain what happens on cache miss, cache outage, stale data, and partial eviction pressure.

## Rules

- Every cache-backed workflow needs a credible fallback path.
- Degraded mode should preserve the most important business journey, not every convenience feature.
- Staleness promises must be visible to application owners and operators.
- A fast cache that fails unpredictably is worse than a slower system with explicit fallback behavior.

## Degraded-Mode Heuristics

### Define what the user can lose safely

When Redis is slow or unavailable, teams should know:

- which data can be recomputed
- which features can be temporarily simplified
- which user journey must remain trustworthy
- which latency increase is acceptable for recovery mode

### Separate cache dependency classes

Not all keys deserve equal fallback treatment. Distinguish:

- optional acceleration caches
- user-visible session or coordination state
- expensive computed views
- rate limiting or abuse-control primitives

### Practice miss storms mentally before incidents

The real question is not whether the cache is useful. It is whether the backing system survives when Redis suddenly stops helping.

## Common Failure Modes

### Cache dependency denial

The team calls Redis a cache, but the application behaves as if Redis were a required primary system.

### Stale fallback confusion

The product continues operating, but users cannot tell whether they are seeing stale, partial, or recomputed state.

### Miss storm collapse

Redis degrades, fallback traffic hits the backing store, and the system turns one cache incident into a wider platform incident.

## Principal Review Lens

- What happens if this cache returns nothing for ten minutes?
- Which feature can degrade safely and which cannot?
- Does fallback preserve correctness, only availability, or neither?
- What cache dependency would cause the largest secondary incident if Redis slowed down tomorrow?
