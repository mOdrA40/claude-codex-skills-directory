# Failure Stories and Anti-Patterns in Node.js Backends

## Purpose

Rules are easier to ignore than failure stories. This guide captures the kinds of backend failures that repeatedly happen in Node.js systems when teams optimize for shipping speed without production discipline.

## Failure Story: The Friendly Async Handler That Melted the Service

A handler looked simple:

- parse payload
- call one database
- fan out to several downstream services
- aggregate results

It worked in test and small load. Under production traffic:

- `Promise.all` amplified dependency slowness
- retries stacked at multiple layers
- event-loop delay increased because serialization and logging exploded
- memory rose because pending work buffered faster than downstream services recovered

### Lesson

- unbounded concurrency is a production bug
- retries need one owner
- aggregation endpoints should define backpressure and deadlines explicitly

## Failure Story: The Invisible Background Job

A service used `setInterval` and fire-and-forget promises for important work:

- no ownership
- no lag metrics
- no retry taxonomy
- no shutdown handling

During deploys, jobs double-ran and sometimes silently disappeared.

### Lesson

If work matters, it needs explicit lifecycle, observability, and replay semantics.

## Anti-Patterns

### Hidden Global State

- global caches with unclear lifetime
- mutable singletons for clients or config
- request context stored where async boundaries can leak it

### Retry Everywhere

- client retries
- SDK retries
- queue retries
- handler retries

This creates incident amplification, not resilience.

### Transport-Led Design

When route handlers become the architecture, business policy, transaction logic, and dependency orchestration end up scattered.

## Review Questions

- what failure story is this design trying to avoid?
- where could latent concurrency become visible only under load?
- what work exists without a clear owner or metric?
