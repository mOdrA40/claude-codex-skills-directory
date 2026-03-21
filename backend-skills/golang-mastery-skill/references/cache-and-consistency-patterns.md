# Cache and Consistency Patterns in Go Services

## Principle

Caching improves latency until it quietly breaks correctness, tenant isolation, or incident response. A principal-level Go service treats cache behavior as a consistency policy, not a speed trick.

## Rules

- define source of truth explicitly
- choose cache-aside, write-through, or write-behind intentionally
- bound staleness expectations
- make cache keys scope-aware for tenant and version boundaries
- decide how the service behaves when cache is unavailable

## Bad vs Good

```text
❌ BAD
Cache behavior is added ad hoc and stale reads are discovered only after incidents.

✅ GOOD
Freshness, invalidation, and degraded-mode behavior are explicit.
```

## Review Questions

- what stale data is acceptable?
- does cache failure break correctness or only latency?
- are invalidation and schema/version boundaries clear?
