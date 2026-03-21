# Context Propagation and Async Boundaries

## Rules

- Propagation failures destroy the value of distributed tracing faster than missing spans.
- HTTP, gRPC, messaging, batch jobs, and background work each need explicit boundary handling.
- Sampling, baggage, and trace context should be understood across service hops.
- Async workflows must preserve causality even when timing and execution ownership change.

## Common Failure Modes

- One middleware layer dropping headers or creating new roots silently.
- Message consumers losing context because payload and transport handling diverge.
- Mixed libraries or legacy code paths breaking propagation across only some flows.
- Teams trusting traces without validating end-to-end propagation coverage.

## Principal Review Lens

- Which boundary is most likely to break propagation under real load?
- Can the team verify propagation across sync and async paths quickly?
- Are we confusing baggage with labels or auth context dangerously?
- What would make a trace look healthy while still being wrong?
