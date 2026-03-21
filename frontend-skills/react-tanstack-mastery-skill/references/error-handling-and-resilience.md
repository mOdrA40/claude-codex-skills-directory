# Error Handling and Resilience in React Applications

## Purpose

Resilient UIs distinguish loading, empty, stale, degraded, retryable, and fatal states clearly. Many frontend bugs are actually state-modeling failures.

## Rules

- model loading/error/empty states intentionally
- use route-level and feature-level error boundaries appropriately
- distinguish mutation errors from query display errors
- keep retry behavior explicit
- prefer recoverable degraded UI over full-page failure when safe

## Failure-State Heuristics

### Recoverable view failure

Examples:

- one panel fails but the page is still useful
- secondary data fails to load
- a retry can recover without full navigation reset

### Route-level failure

Examples:

- critical loader fails
- required auth or session state is invalid
- the route cannot render meaningful content safely

### Mutation failure

This should often be treated differently from route rendering failure because the user may still keep working while retrying or correcting the action.

## Common Failure Modes

### Permanent loading masquerading as resilience

The UI never exits loading because the failure path was not modeled.

### Boundary placement that hides ownership

Error boundaries exist, but no one knows whether recovery should happen at widget, feature, or route level.

### Retry spam

The app retries too aggressively and turns a dependency incident into a worse user experience.

## Review Questions

- what does the user see when a dependency is slow?
- can the page recover without full reload?
- is stale-but-usable data treated differently from hard failure?
- which failures deserve degraded mode instead of full error replacement?
