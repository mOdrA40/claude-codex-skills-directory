# Incident Runbooks (Redis)

## Rules

- Cover eviction storms, memory exhaustion, failover issues, hot keys, and cache stampede.
- Stabilize user-facing correctness before optimizing latency.
- Include safe emergency actions and forbidden commands.
- Tie recovery to observable metrics and app behavior.

## Incident Classes

Redis runbooks should explicitly separate:

- memory pressure and eviction storms
- hot key or big key incidents
- failover or replication issues
- cache stampede and miss-storm behavior
- coordination or session-state failures where Redis behaves more like infrastructure than cache

## Operator Heuristics

### Classify dependency type first

Before acting, determine whether Redis is currently serving as:

- optional acceleration
- critical session or coordination state
- rate limiter / control plane primitive
- queue or stream backbone

### Protect correctness first

Fast but wrong fallback behavior can be worse than temporary latency pain if users see corrupt, stale, or contradictory state.

## Common Failure Modes

### Cache-only mental model

The team responds as if Redis were optional even though some product paths now depend on it for correctness or coordination.

### Temporary hit-rate recovery mistaken for safety

Metrics improve briefly, but fallback load, stale state, or eviction churn still threaten wider system stability.

## Principal Review Lens

- Can on-call stop cascading misses quickly?
- Which action risks data loss or wider outage?
- What confirms true recovery instead of temporary calm?
- Which Redis incident path would currently create the biggest secondary outage in dependent systems?
