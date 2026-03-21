# Hono API Patterns for Production

## Why Hono Works Well

Hono encourages a Web-standard style that travels well across Bun, Node.js, and edge-like runtimes. That portability is useful only if the service still has clear boundaries and stable operational behavior.

## Rules

- keep route handlers thin
- validate at the edge
- centralize error mapping
- keep request context enrichment explicit
- do not bury retries or transactions inside handlers

## Bad vs Good

```typescript
// ❌ BAD: handler mixes parsing, DB work, and dependency orchestration.
app.post('/orders', async (c) => {
  const body = await c.req.json()
  const order = await db.orders.insert(body)
  await paymentClient.charge(order)
  return c.json(order)
})
```

```typescript
// ✅ GOOD: handler validates and delegates.
app.post('/orders', validator('json', orderSchema), async (c) => {
  const input = c.req.valid('json')
  const order = await createOrder.execute(input)
  return c.json(order, 201)
})
```

## Middleware Guidance

Use middleware for:

- request IDs
- authn/authz attachment
- logging/tracing
- body limits
- error mapping

Do not use middleware to hide core business policy.

## Operational Questions

- Does every handler map errors consistently?
- Are body limits enforced?
- Are outbound calls timed out and observable?
- Can the service drain cleanly on shutdown?
