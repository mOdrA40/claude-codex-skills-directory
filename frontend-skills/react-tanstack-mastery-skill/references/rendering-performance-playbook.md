# Rendering and Performance Playbook for React

## Principle

Performance work should target actual bottlenecks: rendering frequency, list cost, network waterfalls, and bundle boundaries. Memoization without measurement is noise.

## Investigation Order

1. identify slow route or interaction
2. inspect render frequency
3. inspect network waterfall and cache behavior
4. inspect list/table cost
5. inspect bundle split and hydration cost

## Common Failure Classes

- query waterfalls
- over-broad invalidation
- unstable props breaking memoization
- oversized lists without virtualization
- expensive derived calculations on every render

## Investigation Heuristics

### Slow initial route load

Look at:

- route-level code splitting
- duplicated loaders
- large initial bundles
- blocking data dependencies

### Slow interactions after load

Look at:

- component tree re-render breadth
- table/list rendering cost
- unnecessary derived computations
- over-shared state updates

### Feels slow but profiler is unclear

Look at the network layer too. Often the bottleneck is stale cache configuration, duplicated fetches, or mutation invalidation cascades rather than raw rendering time.

## Bad vs Good

```text
❌ BAD
Wrap everything in `useMemo` and `useCallback` without knowing the bottleneck.

✅ GOOD
Measure first, then optimize the path that actually hurts the user experience.
```

## Review Questions

- is the bottleneck network, render, or bundle?
- is URL state causing redundant work?
- are tables/forms re-rendering more than needed?
- what measurable user-facing metric will improve if this optimization is correct?
