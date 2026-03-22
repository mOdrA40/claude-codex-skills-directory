# High-Risk Change Preflight Checklist for Node.js Services

## Use This Before

- schema or contract changes
- queue semantics changes
- retry policy changes
- auth or security changes
- major dependency upgrades
- rollout-sensitive configuration changes

## Checklist

- define rollback trigger
- confirm observability for new failure mode
- confirm mixed-version compatibility
- confirm blast radius and containment plan
- confirm on-call can diagnose the change
- confirm load or backlog impact is understood
