# KMP Architecture and Shared Boundaries

## Principle

Shared code is valuable only when it reduces real duplication without hiding platform ownership, lifecycle differences, or release complexity.

## Shared-Code Heuristics

Share code when it improves consistency in:

- domain rules
- validation
- networking contracts
- repository abstractions
- sync policies
- analytics event definitions

Do not share code blindly when platform behavior meaningfully differs.

## Keep Platform Ownership Explicit

Android and iOS may differ in:

- navigation model
- lifecycle timing
- permission flows
- background execution rules
- local storage implementation
- release and rollback behavior

KMP should not erase those differences artificially.

## Common Failure Modes

### Shared-module vanity

Teams move code into shared modules to maximize percentage of shared lines instead of reducing real maintenance cost.

### Leaky abstractions

Platform-specific lifecycle or UI assumptions sneak into shared code, making both sides harder to evolve.

### Release coupling

A shared-module change affects both platforms, but rollout and compatibility strategy were never planned.

## Bad vs Good

```text
❌ BAD
Force navigation, persistence, and UI-centric state through shared code just to increase KMP coverage.

✅ GOOD
Share domain and sync logic where it reduces real duplication, while keeping platform UX and lifecycle ownership explicit.
```

## Review Questions

- what real duplication does shared code remove?
- what platform-specific behavior is being hidden by the abstraction?
- what is the compatibility story when shared persistence or API contracts change?
