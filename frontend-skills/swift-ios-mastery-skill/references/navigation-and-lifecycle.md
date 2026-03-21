# Navigation and App Lifecycle

## Rules

- model app lifecycle transitions explicitly
- avoid mixing navigation, async loading, and business rules into one view layer
- keep deep links and auth transitions deterministic

## Lifecycle Pressure Points

Important moments include:

- cold start
- foreground to background transitions
- push notification wake-ups
- deep-link entry
- session expiration while app is open

Each of these can invalidate naive navigation assumptions.

## Common Failure Modes

### Navigation driven by side effects only

The app redirects because of scattered async checks, making it hard to understand why users landed on the current screen.

### Deep-link partial state

The app can open a target route, but supporting data or auth state is not ready yet.

### Resume mismatch

The visible screen and the underlying session or persistence state no longer agree after resume.

## Review Questions

- what happens if the app is resumed halfway through a sensitive flow?
- which lifecycle events can invalidate current navigation assumptions?
- can a deep link land safely before all supporting state is restored?
