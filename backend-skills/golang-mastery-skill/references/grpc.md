# gRPC (Go Services)

Use gRPC when strong contracts, streaming, and multi-service internal communication bring real value. Do not choose it just because it looks more “enterprise”.

## When gRPC is the right choice

Prefer gRPC when:

- you own both client and server,
- latency matters and JSON overhead is non-trivial,
- you need bi-directional or server streaming,
- schema governance matters across many services.

Prefer HTTP/JSON when:

- public API ergonomics matter more,
- browser clients are primary,
- operational maturity for gRPC is weak,
- simple CRUD over the internet is the main use-case.

## Service Design Defaults

- Keep protobuf contracts stable and versioned.
- Do not expose storage schema directly through proto messages.
- Make error mapping explicit: invalid, not found, conflict, rate-limited, unavailable.
- Use deadlines on every client call.
- Propagate trace context and request identity across boundaries.

## Streaming Guardrails

Streaming is where systems quietly become expensive.

- Every stream needs cancellation handling.
- Bound message sizes and backpressure behavior.
- Define server-side resource limits per stream.
- Decide what happens on partial delivery, reconnect, and duplicate consumer behavior.

## Retries and Idempotency

- Do not blindly retry non-idempotent RPCs.
- Idempotency keys still matter for create/payment-like flows.
- Keep retry policy centralized and deadline-aware.
- Avoid retry amplification across caller, service mesh, and worker layers.

## Security Defaults

- Use mTLS or another strong service-to-service auth strategy.
- Keep authn/authz outside of handlers where possible, but close to the boundary.
- Validate tenant/account context explicitly.
- Treat internal RPCs as hostile enough to validate inputs anyway.

## Operational Checklist

- Health endpoints and readiness still matter even for gRPC servers.
- Expose request/error/duration metrics per method.
- Log stable method names, status codes, and trace IDs.
- Test behavior under deadline exceeded, cancellation, and dependency failure.

## Principal Review Lens

Ask:

- Is gRPC solving a real problem or introducing accidental complexity?
- What is the migration story when the proto contract changes?
- How are deadlines, retries, and idempotency enforced?
- What happens when a stream stalls or a client disconnects mid-flight?
