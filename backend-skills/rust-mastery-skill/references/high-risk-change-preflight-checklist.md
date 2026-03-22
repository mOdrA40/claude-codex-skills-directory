# High-Risk Change Preflight Checklist for Rust Services

## Use This Before

- schema or contract changes
- async model changes
- retry policy changes
- queue semantics changes
- runtime or dependency upgrades with production impact

## Checklist

- define rollback trigger
- confirm mixed-version compatibility
- confirm observability for new failure modes
- confirm blast radius and containment plan
- confirm queue, pool, and dependency impact is understood
