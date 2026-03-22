# Service Decomposition and Boundary Decisions in Go Backends

## Core Principle

Choose new service boundaries when they improve ownership, scaling, or failure isolation enough to justify contract and deployment complexity.

## Questions

- does this split remove a real bottleneck?
- does it reduce blast radius or just add network hops?
- can a modular monolith solve the same problem more safely?
- can the team support added contracts and on-call paths?

## Principal Heuristics

- Splitting too early creates distributed systems tax before distributed systems value.
- Service boundaries should align with ownership and failure semantics.
