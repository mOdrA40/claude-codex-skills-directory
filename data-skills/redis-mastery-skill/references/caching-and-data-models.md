# Caching and Data Models (Redis)

## Caching Rules

- Define cache-aside, write-through, or write-behind explicitly.
- Every key needs ownership and TTL strategy.
- Plan for cache stampede and cold-start behavior.
- Avoid storing unbounded sets or giant payloads casually.

## Data Model Rules

- Choose strings, hashes, sets, sorted sets, streams, and bitmaps intentionally.
- Model key names for discoverability and operational safety.
- Hot keys need explicit mitigation.

## Design Heuristics

### Cache strategy should reflect correctness risk

Cache-aside, write-through, and write-behind are not interchangeable convenience patterns. Each changes staleness behavior, failure handling, and downstream load in different ways.

### Data structure choice shapes operability

The best Redis data model is usually the one that keeps memory behavior, mutation cost, and failure recovery easiest to reason about under load.

### Cold-start behavior should be part of the design

If Redis were empty right now, teams should know which paths degrade gracefully, which become expensive, and which user journey must remain trustworthy.

## Common Failure Modes

### Cache pattern by habit

The team uses a familiar caching style without proving it matches the freshness and recovery needs of the workload.

### Data structure mismatch

One Redis structure is chosen because it is convenient, but its memory growth, mutation cost, or debugging behavior makes incidents worse later.

### Empty-cache panic

The system depends on warm cache state more heavily than its owners admit, so one restart or eviction wave becomes a wider platform event.

## Principal Review Lens

- What happens if Redis is empty right now?
- Which keys can grow without bound?
- Is invalidation strategy actually correct under concurrency?
- Which caching assumption is most likely to fail under cold start or widespread eviction?
