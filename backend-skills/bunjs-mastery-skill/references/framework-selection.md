# Bun Framework Selection: Hono vs Elysia vs Minimal HTTP

## Decision Matrix

### Choose Hono when

- portability across runtimes matters
- Web API alignment is valuable
- you want small, explicit handlers
- edge-like deployment or future portability is likely

### Choose Elysia when

- deep TypeScript ergonomics and framework conventions improve team speed
- schema-first DX is valuable
- the team accepts tighter framework coupling

### Choose minimal HTTP primitives when

- the service is very small
- the team wants maximum explicitness
- framework abstraction would add little value

## Principal Guidance

Do not choose based on benchmarks alone.

Choose based on:

- error handling discipline
- portability needs
- framework lock-in tolerance
- team familiarity
- debugging ergonomics
- plugin complexity

## Bad vs Good

```text
❌ BAD
Pick the fastest framework and then improvise architecture later.

✅ GOOD
Pick the framework whose constraints match the service shape, team capability, and operational expectations.
```
