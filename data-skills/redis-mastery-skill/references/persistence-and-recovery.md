# Persistence and Recovery (Redis)

## Rules

- Pick RDB, AOF, or both based on recovery objectives.
- Persistence settings are business decisions, not defaults.
- Test restart behavior and data loss windows.
- Do not promise durability beyond configured reality.

## Recovery Heuristics

### Match persistence mode to business truth

If the application treats Redis as durable enough to matter during restart or failover, persistence settings should be chosen and explained like an explicit recovery contract.

### Startup behavior is part of reliability

Recovery quality depends not only on what data survives but also on how long restart, replay, warm-up, and downstream stabilization take.

### Be honest about accidental primary-store behavior

Many teams say Redis is just a cache while operational reality shows that important session, coordination, or derived state cannot be lost casually.

## Common Failure Modes

### Durability theater

Teams speak about Redis as if it were durable enough for business-critical state without validating actual crash-loss windows and replay behavior.

### Restart optimism

A node comes back, but warm-up cost, repopulation pressure, and downstream dependency load create a second incident.

### Persistence without restore realism

The config looks responsible, but nobody has tested realistic recovery timing or data-quality expectations after failure.

## Principal Review Lens

- What data loss is possible on crash right now?
- How long is restart and warm-up under realistic memory size?
- Are teams using Redis as durable storage by accident?
- What recovery promise are we implicitly making that Redis is not actually configured to keep?
