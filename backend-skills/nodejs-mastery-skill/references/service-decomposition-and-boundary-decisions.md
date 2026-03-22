# Service Decomposition and Boundary Decisions in Node.js Backends

## Core Principle

Do not split services because the codebase feels emotionally large. Split when operational or domain boundaries justify independent failure, scaling, or ownership.

## Questions

- does this boundary reduce blast radius or just create more network hops?
- does independent scaling actually matter here?
- can the team operate additional contracts, rollouts, and incidents?
- would a modular monolith solve the problem more safely?

## Principal Heuristics

- Prefer a modular monolith until split pressure is real.
- Boundary mistakes create long-term operational tax.
