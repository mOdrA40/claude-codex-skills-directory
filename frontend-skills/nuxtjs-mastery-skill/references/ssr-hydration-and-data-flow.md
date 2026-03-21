# SSR, Hydration, and Data Flow in Nuxt 3

## Principle

Nuxt apps fail in subtle ways when server data flow, client hydration, and browser-only assumptions are mixed carelessly.

## Rules

- know what runs on server vs client vs both
- keep hydration-safe defaults explicit
- avoid browser-only side effects before mount
- make async data boundaries predictable

## Common Failure Modes

- hydration mismatch from non-deterministic rendering
- duplicated fetches across server and client
- composables that assume browser APIs too early
- state mismatch from implicit side effects

## Practical Heuristics

### Keep first render deterministic

If the server renders one thing and the client immediately decides something else, hydration trust is already broken.

### Know which data is route-blocking

Not every async dependency deserves to block first render. Choose intentionally:

- critical SEO or above-the-fold data
- user-specific data that can wait
- analytics or enhancement data that should definitely wait

### Browser-only logic must be deliberate

Feature detection, local storage access, viewport logic, and non-deterministic values should not accidentally affect initial server markup.

## Incident Patterns

### Only production has hydration errors

Often caused by cache behavior, environment differences, or data timing that local development did not reproduce.

### One route class regresses badly

Usually means rendering strategy and data dependency shape were not chosen per route class.

## Review Questions

- is this data fetched once or twice?
- what assumptions differ between server and client?
- can the route render deterministically on first paint?
- what can safely defer until after hydration?
