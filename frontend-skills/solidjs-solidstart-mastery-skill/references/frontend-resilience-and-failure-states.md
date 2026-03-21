# Frontend Resilience and Failure States in Solid Apps

## Purpose

Fine-grained reactivity does not remove the need to model degraded UI states well. Users still need trustworthy behavior when data is slow, stale, partial, or broken.

## Rules

- distinguish loading, partial, stale, and failed views
- keep suspense and error boundaries intentional
- avoid hiding failed async work behind permanent loading states
- model user recovery paths clearly

## Failure-State Categories

### Partial failure

One resource or panel fails, but the rest of the route can remain useful.

### Route-critical failure

The route cannot safely or meaningfully render without specific data.

### Recoverable interaction failure

An action fails, but the view should remain interactive while the user retries or adjusts input.

## Common Failure Modes

### Suspense as camouflage

Suspense boundaries make the app feel neat, but they can also hide ownership problems where async failure was never modeled explicitly.

### Error handling too high or too low

Either one failure destroys too much UI, or error handling is so local that recovery becomes inconsistent.

### Stale state without truth signaling

The user sees data, but cannot tell whether it is fresh, partial, or failed to update.

## Review Questions

- what does the user see if a resource partially fails?
- does the view recover on retry without reloading the whole app?
- are suspense boundaries placed where they help or where they hide complexity?
- which failures should degrade gracefully instead of replacing the entire route?
