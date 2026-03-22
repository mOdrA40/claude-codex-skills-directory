# Service Decomposition and Boundary Decisions in Rust Backends

## Core Principle

Split services when failure isolation, ownership, or scaling semantics justify the operational tax.

## Questions

- does this split reduce blast radius or just add hops?
- can a modular monolith solve the same problem more safely?
- can the team operate extra rollouts, contracts, and incidents?
- is independent scaling truly needed?

## Principal Heuristics

- Distributed systems tax should be paid only when the value is real.
- Boundaries should follow failure and ownership semantics.
