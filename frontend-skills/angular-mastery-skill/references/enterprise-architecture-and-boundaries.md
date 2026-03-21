# Enterprise Architecture and Boundaries in Angular

## Principle

Angular apps often live for years and across teams. The architecture must reduce organizational entropy, not only local component complexity.

## Boundary Model

Separate:

- feature areas
- shared UI primitives
- domain or orchestration services
- platform services
- route ownership
- state and async ownership

## Common Failure Modes

- shared modules becoming dumping grounds
- services that act as hidden global state
- architecture that mirrors folders instead of ownership and workflow reality

## Review Questions

- which boundary reduces organizational conflict?
- what will become a shared dumping ground if left vague?
- which service currently acts as an architectural escape hatch?
