# Bun Backend Architecture Decision Framework

## Principle

Bun makes it easy to start fast, but principal-level architecture is still about choosing the right boundaries for reliability, portability, and operability.

## Choose a Small Explicit Service When

- one API surface and few dependencies exist
- Hono or minimal HTTP primitives are sufficient
- operational complexity should stay minimal

## Choose Stronger Layering When

- database plus queue plus external APIs must be coordinated
- idempotency and retries matter
- multiple engineers will evolve the service
- you need strict error mapping and boundary control

## Do Not Overfit to Benchmarks

A runtime benchmark does not answer:

- how incidents are debugged
- how contracts evolve
- how migrations are deployed safely
- how background work is owned

## Review Questions

- which architecture reduces future accidental complexity?
- what failure mode becomes easier with this boundary choice?
- which runtime or framework assumption are we hard-coding into the design?
