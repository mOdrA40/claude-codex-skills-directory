# High-Risk Change Preflight Checklist for Go Services

## Use This Before

- schema or contract changes
- concurrency model changes
- retry policy changes
- queue semantics changes
- dependency upgrades with production impact

## Checklist

- define rollback trigger
- confirm mixed-version compatibility
- confirm observability for new failure modes
- confirm blast radius and containment plan
- confirm load, pool, and backlog impact is understood
