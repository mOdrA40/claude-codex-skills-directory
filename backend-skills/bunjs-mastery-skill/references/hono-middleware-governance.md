# Hono Middleware Governance

## Principle

Middleware should simplify cross-cutting concerns, not become a hidden architecture where business behavior is scattered and impossible to debug.

## Rules

- keep middleware for auth wiring, request IDs, tracing, body limits, and error mapping
- avoid putting domain decisions into middleware
- document ordering dependencies
- keep request mutation explicit and minimal

## Review Questions

- would an incident responder understand middleware order quickly?
- is business policy hidden in middleware instead of services?
- can handlers be reasoned about without reading five middleware layers first?
