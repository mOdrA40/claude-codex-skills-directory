# RxJS and Async Discipline in Angular

## Principle

RxJS is powerful when stream ownership is explicit and dangerous when teams use it to hide lifecycle and state complexity.

## Rules

- know who subscribes and who completes
- model stream lifecycles intentionally
- avoid giant chains that obscure business meaning
- distinguish stream composition from application state ownership

## Stream Design Heuristics

### Streams should express domain or UI meaning

A stream should be understandable in product terms, not just operator terms like “combine this observable with that observable because it works.”

### Lifecycle must be visible

Teams should be able to answer:

- when does this stream start?
- when does it end?
- who owns cleanup?
- what happens on route change or component destruction?

### Keep state and transport concerns distinct

Not every async stream should become the main source of truth for UI state.

## Common Failure Modes

- leaked subscriptions
- stream graphs no one can explain during incidents
- async behavior hidden inside services that appear simple

### Stream superstition

RxJS is used because it is powerful, not because the specific problem needs that level of composition.

### Recovery ambiguity

The app has complex stream behavior, but no one knows how errors, retries, or degraded states should appear to users.

## Review Questions

- what business meaning does this stream model?
- where does lifecycle end?
- would a simpler async model be clearer?
- what error or retry behavior would be hardest to explain during an incident?
