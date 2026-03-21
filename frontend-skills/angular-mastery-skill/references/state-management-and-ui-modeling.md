# State Management and UI Modeling in Angular

## Principle

Angular state should be modeled by ownership and lifecycle, not by whichever state library is fashionable at the moment.

## State Categories

### Local UI state

Examples:

- dialog visibility
- current tab or sort choice
- local form interactions

This should stay near the component or feature boundary unless there is a real reason to elevate it.

### Feature state

Examples:

- route-bound async data
- retry and refresh state
- filters that affect one workflow
- mutation status visible to the user

This should have explicit ownership and predictable reset behavior.

### Durable application state

Examples:

- authenticated session context
- persisted preferences
- shared cross-feature business context

This must be treated as release-sensitive and operationally important state.

## Heuristics

- keep local UI state local
- elevate state only when multiple features truly share it
- distinguish async resource state from durable app state
- avoid services that silently become state stores without clear contracts

## Common Failure Modes

- god services
- duplicated async state across component and service layers
- poor distinction between route state, UI state, and durable application state

### State escalation by convenience

State moves into shared services not because multiple features truly need it, but because local ownership feels harder in the short term.

### Recovery state blindness

The app models happy paths but not stale, retrying, partial, or degraded states clearly.

## Review Questions

- where does this state begin and end?
- who is allowed to mutate it?
- can recovery states be rendered predictably?
- which state is currently shared only because no one wanted to decide ownership precisely?
