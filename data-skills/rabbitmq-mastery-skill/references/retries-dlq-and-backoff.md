# Retries, DLQ, and Backoff

## Rules

- Retry policy must avoid hot loops and silent backlog growth.
- DLQ ownership and replay workflow must be explicit.
- Exponential backoff should reflect business urgency and system safety.
- Poison messages need quarantine, not wishful retries.

## Recovery Heuristics

### Retries are load-shaping decisions

Every retry policy changes queue depth, downstream pressure, operator visibility, and how quickly one local failure becomes a platform-wide incident.

### DLQ is an unresolved business problem

Routing a message to dead-letter state is only useful if ownership, investigation, replay, and closure are all explicit and practiced.

### Backoff should match dependency reality

Fast retries may be correct for short transient faults, but they are dangerous when downstream systems are rate-limited, partially down, or semantically unable to absorb replays safely.

## Common Failure Modes

### Retry storm by configuration

The system follows configured behavior correctly, but that behavior amplifies load and delay until one incident cascades through the platform.

### DLQ theater

Messages are kept away from the hot path, yet nobody is accountable for what they mean or how recovery should happen.

### Poison-message ambiguity

Teams cannot quickly distinguish transient failure from truly bad payloads, so retries and backoff waste time and capacity.

## Principal Review Lens

- Who owns DLQ triage?
- What retry setting turns one failure into a platform incident?
- How is replay made safe for side effects?
- Which backoff policy looks cautious but still creates dangerous backlog or duplicate risk?
