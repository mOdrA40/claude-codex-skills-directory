# Core NATS Pub-Sub and Request-Reply

## Rules

- Use core NATS when ephemeral, low-latency delivery semantics are acceptable.
- Request-reply should remain bounded by timeout and ownership expectations.
- Distinguish fire-and-forget signaling from workflows that require durable guarantees.
- Simplicity is the strength of core NATS; do not project JetStream semantics onto it.

## Design Guidance

- Know when no subscriber is an acceptable outcome.
- Use request-reply for operationally simple interactions, not distributed transaction fantasies.
- Make timeout behavior and retry policy explicit.
- Keep service discovery and health implications visible.

## Principal Review Lens

- Why is ephemeral delivery correct here?
- What failure mode is acceptable and what is not?
- Are we using request-reply to avoid designing a real contract?
- Which workflow secretly needs durability instead?
