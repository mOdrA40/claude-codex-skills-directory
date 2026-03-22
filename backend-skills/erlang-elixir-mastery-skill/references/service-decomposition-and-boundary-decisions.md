# Service Decomposition and Boundary Decisions on the BEAM

## Core Principle

Split services when failure isolation, ownership, or scaling semantics justify the operational tax of more contracts, nodes, and incidents.

## Questions

- does this split reduce subtree or regional blast radius?
- can a modular monolith or OTP boundary solve this more safely?
- can the team operate added releases, consumers, and contracts?
- is independent scaling actually required?

## Principal Heuristics

- OTP structure is already a strong boundary tool; do not ignore it.
- Distributed boundary decisions should follow ownership and failure semantics.
