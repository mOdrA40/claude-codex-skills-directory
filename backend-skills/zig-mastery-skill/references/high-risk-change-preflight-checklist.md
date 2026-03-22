# High-Risk Change Preflight Checklist for Zig Services

## Use This Before

- schema or contract changes
- allocator or concurrency model changes
- retry policy changes
- worker semantics changes
- dependency upgrades with production impact

## Checklist

- define rollback trigger
- confirm mixed-version compatibility
- confirm observability for new failure modes
- confirm blast radius and containment plan
- confirm memory, queue, and dependency impact is understood
