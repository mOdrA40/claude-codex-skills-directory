# Cache Invalidation (Redis)

## Rules

- Every cache requires a correctness strategy, not just speed goals.
- TTL-only invalidation is often insufficient for critical data.
- Prefer explicit ownership of who writes, invalidates, and repopulates keys.
- Cache stampede protection should be designed, not improvised.

## Principal Review Lens

- What stale read is acceptable and for how long?
- Who owns invalidation correctness?
- What happens during cold cache recovery under peak traffic?
