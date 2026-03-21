# iOS Architecture and State Ownership

## Principle

iOS architecture becomes unstable when teams blur the line between rendering, app state, persistence, and platform services. SwiftUI makes composition easier, but it does not remove the need for strong ownership boundaries.

## Rules

- separate view rendering from domain and persistence logic
- keep observable state boundaries explicit
- avoid god view models
- define what survives app restart and what is ephemeral
- keep platform services outside the rendering layer whenever possible

## State Categories

### Ephemeral view state

Examples:

- sheet visibility
- selection state
- local input drafts that do not need persistence

This usually belongs close to the view.

### Screen or feature state

Examples:

- async loading state
- refresh state
- search/filter state
- user-visible retry state

This should have a clear feature owner and predictable lifecycle.

### Durable app state

Examples:

- authentication session
- local cache or persisted preferences
- offline drafts
- migration-sensitive local data

This must be versioned and handled as release-sensitive state.

## Common Failure Modes

### God view model

One observable object owns navigation, fetching, analytics, persistence, and business rules. It becomes impossible to reason about lifecycle and test failures in isolation.

### Accidental persistence

Temporary UI state gets stored alongside durable state because the boundary was never defined.

### Platform leakage

Push notifications, camera, or background lifecycle assumptions creep directly into view code.

## Review Questions

- which state should disappear when the screen disappears?
- which state must survive app termination?
- what breaks first if persistence migration fails?
- where do platform-service side effects live today?
