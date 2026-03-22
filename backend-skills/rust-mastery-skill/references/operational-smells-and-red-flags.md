# Operational Smells and Red Flags in Rust Services

## Red Flags

- detached tasks without owner or cancellation path
- queue or channel growth with no hard cap
- retry layering across already saturated dependencies
- rollout-sensitive changes with no mixed-version plan
- one tenant or workload monopolizing pools or locks
- unsafe expansion during incident response

## Agent Review Questions

- where does pressure accumulate first: locks, allocators, channels, pools, or downstream timeouts?
- does this change widen blast radius across tenants or workers?
- can operators distinguish rollout regression from dependency failure?
- what signal should stop rollout immediately?

## Principal Heuristics

- Type safety does not prevent operational fragility.
- If buffering hides pressure, failure will arrive later and worse.
- If rollback requires heroics, rollout design is weak.
