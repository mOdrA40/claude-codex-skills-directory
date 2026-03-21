# State Boundaries and Component Architecture in SvelteKit

## Principle

Svelte simplicity is a strength only if state ownership remains clear. Hidden shared state can make a clean component tree deceptive.

## Heuristics

- keep local interaction state local
- keep route data ownership with routes
- use stores when cross-cutting state is real, not because it feels convenient
- keep backend integration concerns out of presentational components

## Common Failure Modes

- overusing stores for state that should remain route-local
- components that mix presentation, fetch logic, and navigation effects
- state that becomes difficult to reset across route transitions

## Review Questions

- which state should reset on navigation?
- which state must survive route boundaries?
- what component owns too much behavior?
