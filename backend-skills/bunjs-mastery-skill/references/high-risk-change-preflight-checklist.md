# High-Risk Change Preflight Checklist for Bun Services

## Use This Before

- migration changes
- retry policy changes
- queue semantics changes
- major framework or runtime upgrades
- rollout-sensitive config flips

## Checklist

- define rollback trigger
- confirm compatibility during mixed-version deploy
- confirm observability for new failure modes
- confirm blast radius and containment plan
- confirm queue and dependency impact is understood
