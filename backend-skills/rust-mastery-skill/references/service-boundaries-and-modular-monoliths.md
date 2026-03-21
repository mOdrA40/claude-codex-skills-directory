# Service Boundaries and Modular Monoliths in Rust

## Principle

Rust teams often have the tools to make highly structured systems early. That does not mean every boundary should become a crate or a service.

## Rules

- prefer modular monoliths until operational reasons justify splits
- align crate boundaries with ownership and compilation value, not aesthetics
- separate transport, orchestration, domain, and adapters clearly
- avoid premature trait forests where simple concrete dependencies would be clearer

## Review Questions

- which boundary reduces operational risk today?
- is a new crate or service lowering coupling or just increasing coordination cost?
- can the domain stay testable without framework types?
