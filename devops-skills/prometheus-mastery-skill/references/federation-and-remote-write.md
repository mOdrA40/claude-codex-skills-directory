# Federation and Remote Write

## Rules

- Federation and remote write solve different problems and should not be conflated.
- Use local Prometheus for low-latency operational queries and remote systems for durable, aggregated, or cross-cluster analysis when needed.
- Remote write fanout, retries, and backpressure can become part of your failure model.
- Multi-cluster design should preserve ownership and query clarity.

## Tradeoffs

- Federation can simplify local autonomy but complicate global views.
- Remote write centralizes history but introduces dependency chains and egress cost.
- Downsampling and long-term storage change what questions can still be answered later.
- Cross-region architectures must account for latency, data sovereignty, and blast radius.

## Principal Review Lens

- What question requires federation versus remote storage?
- What breaks operationally if the remote destination is slow or unavailable?
- Are teams depending on a central observability system for basic cluster debugging?
- Which architecture choice has the lowest blast radius during platform incidents?
