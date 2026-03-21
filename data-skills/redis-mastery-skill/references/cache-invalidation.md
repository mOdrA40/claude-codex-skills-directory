# Cache Invalidation (Redis)

## Rules

- Every cache requires a correctness strategy, not just speed goals.
- TTL-only invalidation is often insufficient for critical data.
- Prefer explicit ownership of who writes, invalidates, and repopulates keys.
- Cache stampede protection should be designed, not improvised.

## Invalidation Heuristics

### Pick invalidation based on correctness risk

Some data can tolerate time-based staleness. Other data needs event-driven invalidation, versioning, or explicit recomputation ownership.

### Invalidation ownership must be singular enough to audit

If multiple services can repopulate or invalidate the same key without a clear contract, the cache becomes fast but untrustworthy.

### Cold-cache behavior matters as much as steady-state behavior

An invalidation strategy is incomplete if the system cannot survive repopulation under peak or degraded conditions.

## Common Failure Modes

### TTL theater

The team uses TTL because it is easy, even though the business surface actually needs explicit correctness control.

### Dual-writer confusion

Two or more producers can refresh or invalidate the same cache path, and no one can explain what state wins under race conditions.

### Stampede after correctness fix

The team fixes stale data with more aggressive invalidation but creates a backend overload problem during refill.

## Principal Review Lens

- What stale read is acceptable and for how long?
- Who owns invalidation correctness?
- What happens during cold cache recovery under peak traffic?
- Which cache path is currently fastest but least trustworthy?
