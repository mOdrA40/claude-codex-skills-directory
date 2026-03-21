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

## Principal Review Lens

- What happens if Redis is empty right now?
- Which keys can grow without bound?
- Is invalidation strategy actually correct under concurrency?
