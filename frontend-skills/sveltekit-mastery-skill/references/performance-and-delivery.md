# Performance and Delivery in SvelteKit

## Principle

SvelteKit performance should be evaluated through route cost, network dependency shape, hydration cost, and adapter/runtime behavior.

## Common Failure Modes

- leaning on small bundle assumptions while route data latency dominates
- route transitions blocked by heavy client work or poor invalidation behavior
- adapters or deployment platforms changing runtime behavior in surprising ways

## Review Questions

- is the pain in route load, hydration, or interaction?
- what runtime constraint changes delivery behavior?
- what can be simplified before being optimized?
