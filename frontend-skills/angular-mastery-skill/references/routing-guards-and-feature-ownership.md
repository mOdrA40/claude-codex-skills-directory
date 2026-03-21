# Routing, Guards, and Feature Ownership in Angular

## Principle

Routing is where enterprise frontend complexity becomes visible. Guards, lazy loading, and feature modules should express ownership and policy clearly.

## Common Failure Modes

- guards performing hidden business logic
- route trees that reflect historical accidents instead of product flows
- lazy-loading boundaries chosen for bundle superstition instead of ownership or user journeys

## Review Questions

- which routes are policy-heavy vs presentation-heavy?
- what feature ownership boundary does the routing tree encode?
- are guards doing work that belongs elsewhere?
