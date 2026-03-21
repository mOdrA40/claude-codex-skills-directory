# NestJS Boundaries and Architectural Discipline

## Principle

NestJS gives structure quickly, but the same decorators and modules that help teams move fast can also hide poor dependency direction and accidental coupling.

## Rules

- use modules to express domain boundaries, not just folder grouping
- keep controllers thin and providers focused
- avoid turning global modules into hidden service locators
- keep framework decorators from dominating domain logic
- use Nest abstractions where they clarify ownership, not where they add ceremony without boundary value

## Bad vs Good

```text
❌ BAD
A module exports half the application and services call each other freely across domains.

✅ GOOD
Modules map to coherent business capabilities and dependency direction remains obvious.
```

## Review Questions

- is Nest structure reinforcing architecture or hiding weak boundaries?
- are interceptors/guards/filters replacing explicit service design?
- do modules express domain ownership clearly?
