# Server State, Cache, and Sync in Mobile Apps

## Principle

Mobile clients must treat sync as a product concern, not just a fetch concern.

## Rules

- define stale tolerance by screen
- separate optimistic UX from actual sync guarantees
- handle reconnect and retry intentionally
- expose user-visible sync state when correctness matters

## Sync Heuristics

### Fast-moving feed or browse data

This can often tolerate short staleness windows if the UI remains responsive and refresh behavior is clear.

### User-generated or business-critical mutations

These require stronger guarantees:

- clear pending state
- retry ownership
- duplicate submission protection
- explicit failure visibility

### Offline draft workflows

These require a decision about source of truth:

- local-first until sync
- server-authoritative with queued changes
- conflict-aware merge model

## Common Failure Modes

### False optimism

The UI behaves as if a mutation succeeded, but reconnect or backend rejection later invalidates the assumption.

### Invisible sync lag

Data looks current but is actually stale, and the user has no clue whether updates have been sent or received.

### Retry storms

Network reconnect triggers repeated sync attempts without bounding or idempotency.

## Review Questions

- what happens when the user goes offline mid-mutation?
- what data may be stale but usable?
- when should sync failures surface to users?
- which mutations need idempotency or duplicate protection?
- who owns retry policy: screen, service, or sync engine?
