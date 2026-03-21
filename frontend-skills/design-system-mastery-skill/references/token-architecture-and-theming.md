# Token Architecture and Theming

## Principle

Tokens are not just values. They are the contract between design intent, implementation, theming, and multi-product consistency.

## Token Layers

- foundational tokens
- semantic tokens
- component-level tokens where justified
- theme overrides with explicit scope

## Design Heuristics

### Prefer meaning over implementation

Tokens should describe intent such as surface, text, border, emphasis, or status—not incidental implementation details that trap the system in current styling choices.

### Theme boundaries must stay explicit

If product teams can bypass the semantic layer casually, theming becomes a collection of local exceptions instead of a real system capability.

### Component tokens should be earned

Do not create component-specific tokens just because one component exists. Add them when a stable, reusable design distinction actually needs its own contract.

## Common Failure Modes

- tokens named by implementation instead of meaning
- themes that bypass token layers entirely
- component styling decisions escaping system governance

### Token sprawl

The system gains many near-duplicate tokens that no longer express a coherent model.

### Product-local theming drift

Individual teams introduce variant overrides that solve local needs but slowly fracture the platform.

## Review Questions

- what meaning does this token represent?
- which overrides are allowed and which break system integrity?
- how does theming scale across products without chaos?
- which token is likely to become legacy debt if adopted widely today?
