# Compose State and UI Modeling

## Rules

- keep UI state explicit
- separate ephemeral interaction state from durable screen state
- avoid state holders that mix too many unrelated concerns

## State Modeling Heuristics

### Ephemeral interaction state

Examples:

- expanded/collapsed UI
- text field focus
- selected tab
- dialog visibility

This usually belongs close to the composable boundary.

### Screen state

Examples:

- loading and refresh state
- retry state
- visible filters and sort order
- user-visible form validation state

This should be represented explicitly so the UI can render each state predictably.

### Durable or shared state

Examples:

- session state
- offline drafts
- persisted preferences
- sync-sensitive data caches

This should not be confused with small interaction state.

## Common Failure Modes

### Monster UI state holder

One view model or state holder owns navigation, analytics, business rules, persistence, and every screen branch. The result is poor reuse and fragile changes.

### Hidden state transitions

State changes are triggered indirectly by side effects, making recomposition behavior and user-visible recovery difficult to reason about.

## Review Questions

- which state belongs to the composable and which belongs to the feature?
- are loading, empty, failed, and stale states represented explicitly?
- what parts of the screen become hard to test because state ownership is unclear?
