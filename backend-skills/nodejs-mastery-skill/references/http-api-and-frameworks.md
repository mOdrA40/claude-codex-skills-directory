# HTTP APIs and Framework Guardrails

## Framework Choice

Choose frameworks for team fit and operational clarity, not fashion.

- Express: broad ecosystem, simple, but easier to under-structure
- Fastify: strong performance and plugin model, good for disciplined services
- NestJS: convention-heavy, useful for large teams if architectural discipline remains explicit
- Hono: strong Web API alignment, especially attractive for edge/Bun-style portability

## API Rules

- validate at the edge
- map errors consistently
- set request size limits
- use idempotency keys for retried writes
- never expose raw dependency errors

## Bad vs Good

```text
❌ BAD
Every route decides its own status codes and error payloads.

✅ GOOD
One error taxonomy maps consistently across handlers.
```
