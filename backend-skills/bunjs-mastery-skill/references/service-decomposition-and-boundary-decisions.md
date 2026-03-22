# Service Decomposition and Boundary Decisions in Bun Backends

## Core Principle

Split services only when failure isolation, ownership, or scaling differences justify the operational tax.

## Questions

- does this split reduce blast radius?
- does it avoid a real bottleneck or just move it over HTTP?
- can the team operate more deploys, contracts, and incidents?
- would a modular monolith remain safer?

## Principal Heuristics

- Boundaries should reduce ambiguity, not increase it.
- Premature decomposition is expensive operational debt.
