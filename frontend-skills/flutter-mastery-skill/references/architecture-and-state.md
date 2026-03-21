# Flutter Architecture and State Boundaries

## Rules

- separate widget composition from domain logic
- choose state management by scope and volatility, not trend
- keep async data ownership clear
- avoid giant god-notifiers or god-blocs

## State Categories

### Ephemeral UI state

Examples include:

- tab selection
- dialog visibility
- local text input draft
- temporary filter toggles

This should usually remain close to the widget boundary.

### Feature or screen state

Examples include:

- loading and refresh state
- retry state
- screen-specific validation state
- search and sort state

This should have a clear owner and predictable lifecycle.

### Durable application state

Examples include:

- authentication session
- offline cache
- local drafts that survive restarts
- migration-sensitive persisted preferences

This must be handled as release-sensitive state.

## Common Failure Modes

### God notifier or god bloc

One state holder owns fetching, persistence, analytics, navigation decisions, and rendering concerns. This makes change risky and tests noisy.

### Over-shared global state

State is pushed too high because passing boundaries explicitly feels inconvenient. The result is accidental coupling and broad rebuild impact.

### Async ownership confusion

The team cannot answer whether a screen, repository, state holder, or background sync process owns refresh and retry behavior.

## Review Questions

- which state should disappear when the route disappears?
- which state must survive app restart?
- which boundary owns retries and refresh?
- what would break first if persistence migrations fail after release?
