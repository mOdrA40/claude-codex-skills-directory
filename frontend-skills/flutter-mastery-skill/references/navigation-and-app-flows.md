# Navigation and App Flow Strategy in Flutter

## Rules

- model navigation from product flow first
- keep auth/onboarding/deep-link transitions explicit
- avoid pages that own too many unrelated async concerns

## Flow Categories

### App shell flows

Examples:

- splash and bootstrap
- auth resolution
- home shell and tab structure

These should stay stable and predictable because they shape the whole app lifecycle.

### Business flows

Examples:

- checkout
- profile editing
- approval workflows
- onboarding steps

These should be modeled as product journeys, not just a pile of routes.

### Interrupt-driven flows

Examples:

- deep links
- push notification entry
- session expiration
- forced upgrade prompts

These need deterministic recovery behavior.

## Common Failure Modes

### Route-driven chaos

Navigation grows screen by screen without a clear map of entry points, recovery paths, or guarded transitions.

### Hidden async coupling

A route change implicitly depends on data loading, permission checks, or storage restoration that no one documented clearly.

### Broken resume behavior

The app returns from background or deep link entry in a state that no longer matches the visible route stack.

## Review Questions

- what are the critical product flows that must never become ambiguous?
- how does the app recover when launched from a deep link into partial state?
- where are route guards or redirects making hidden business decisions?
