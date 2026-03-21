# Incident Runbooks (Docker)

## Rules

- Cover crash loops, OOMs, bad image rollout, registry failure, and startup regressions.
- Stabilize service first, then optimize image design.
- Include rollback, quarantine, and forensic-friendly steps.
- Tie recovery to measurable runtime and service health signals.

## Principal Review Lens

- Can operators identify bad image versus bad environment fast?
- Which emergency action risks widening the blast radius?
- What confirms true recovery after rollback or restart?
