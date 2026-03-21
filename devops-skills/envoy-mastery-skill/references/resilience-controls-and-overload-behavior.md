# Resilience Controls and Overload Behavior

## Rules

- Circuit breakers, retries, hedging, outlier detection, and overload controls should match workload truth.
- Resilience features can reduce failure or amplify it depending on settings.
- Backpressure and shed-load behavior must be understood before peak incidents.
- Proxy resilience policy should not compensate indefinitely for broken service design.

## Design Guidance

- Tune based on dependency criticality and recovery behavior.
- Align retry budgets with upstream capacity and idempotency guarantees.
- Measure overload behavior under realistic traffic and error conditions.
- Make emergency disable paths explicit.

## Principal Review Lens

- Which resilience control is most likely to worsen an outage?
- Are we using Envoy to hide upstream dysfunction?
- What overload signal should trigger operator action first?
- Can the team explain tradeoffs of current defaults clearly?
