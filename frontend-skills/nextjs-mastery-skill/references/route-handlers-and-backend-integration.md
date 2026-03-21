# Route Handlers and Backend Integration in Next.js

## Principle

Next.js route handlers should not become a hidden micro-backend with unclear ownership. They are powerful, but they need explicit rules.

## Boundary Decision Model

Before placing logic in a route handler, define:

- whether the logic exists for frontend delivery reasons or domain reasons
- whether the runtime target constrains the implementation
- whether auth or session context must be resolved at this boundary
- whether observability and error mapping stay coherent with the backend system of record

## Use Route Handlers For

- request shaping and proxying where frontend deployment needs it
- auth-aware composition of backend responses
- edge-safe or platform-specific request handling
- lightweight integration boundaries that remain operationally understandable

## Use With Extra Caution When

- business orchestration spans multiple downstream systems
- retries or idempotency become important
- payload contracts must stay versioned across clients and services
- incidents require distinguishing frontend proxy logic from backend domain logic quickly

## Avoid Using Them For

- unrelated business orchestration scattered across routes
- complex background workflow ownership
- hidden duplication of backend domain logic

## Common Failure Modes

- route handlers become the easiest place to put everything
- observability and error mapping diverge from core backend services
- teams cannot tell whether a production bug belongs to frontend delivery or backend domain logic

### Proxy boundary drift

The handler started as a thin integration layer and slowly accumulated validation, fallback logic, and business rules until nobody knows where the authoritative behavior lives.

### Runtime mismatch

The code works in one deployment/runtime assumption but behaves differently under edge, node, or platform-specific constraints.

## Review Questions

- why is this logic in Next.js and not in the core backend?
- what happens under deployment/runtime constraints?
- who owns incidents when this boundary fails?
- what route handler is currently acting like a backend service without being treated like one?
