# ElysiaJS Patterns for Production

## Position

ElysiaJS is productive and ergonomic, but the same rule applies as with any fast framework: performance features do not compensate for weak boundaries or undefined failure semantics.

## Rules

- keep schemas explicit
- use typed request contracts at the edge
- separate framework decorators from domain logic
- define one error taxonomy
- document plugin boundaries and ownership

## Bad vs Good

```typescript
// ❌ BAD: framework-specific state leaks into business logic.
app.post('/users', ({ body, set }) => userService.create(body, set))
```

```typescript
// ✅ GOOD: framework input is translated into a domain-friendly contract.
app.post('/users', ({ body }) => createUser.execute(body), {
  body: createUserSchema,
})
```

## Plugin Discipline

Plugins should add:

- auth wiring
- validation
- observability
- dependency injection hooks

Plugins should not become a second hidden architecture.

## Operational Questions

- Are plugin interactions obvious during debugging?
- Are validation failures and domain failures distinct?
- Are background side effects explicit or hidden inside hooks?
