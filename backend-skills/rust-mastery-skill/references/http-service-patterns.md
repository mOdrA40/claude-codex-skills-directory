# HTTP Service Patterns (Rust)

This guide focuses on service boundary discipline for Axum-style APIs and similar Rust HTTP stacks.

## Handler Rules

Handlers should do only this:

- parse and validate input,
- call application logic,
- map result to response,
- attach observability context.

Handlers should not:

- hold business policy,
- compose raw SQL,
- decide retry strategy for dependencies,
- leak framework types into core logic.

## Error Mapping

Define one mapping boundary for public responses.

Recommended categories:

- invalid
- unauthorized
- forbidden
- not_found
- conflict
- rate_limited
- unavailable
- internal

## Request Guardrails

- Bound body size.
- Validate query/path/body explicitly.
- Add request ID / trace context.
- Set timeout expectations at the edge and in dependencies.
- Keep auth and tenant resolution explicit.

## Side Effects

For create/payment/external-callback flows:

- prefer idempotent semantics,
- separate transaction boundary from response mapping,
- use outbox or equivalent when publish-after-write matters.

## Principal Review Lens

Ask:

- Is this handler thin enough?
- Where is input validation guaranteed?
- How are domain errors translated?
- What happens when the downstream dependency is slow or unavailable?
