# Component API Design

## Principle

A component API is a long-term product contract. Convenience today can become years of ambiguity and breaking changes.

## Good API Traits

- explicit intent
- low surprise
- accessible defaults
- predictable composition
- limited but meaningful escape hatches

## API Design Heuristics

### Prefer semantic props over styling trivia

If consumers mostly configure visual implementation details, the component API may be too close to raw CSS and too far from design intent.

### Make invalid combinations hard

The best design-system APIs guide consumers toward safe combinations and make dangerous combinations awkward or impossible.

### Escape hatches must be governed

An escape hatch that bypasses layout, semantics, or accessibility rules too easily will become the default path under product pressure.

## Common Failure Modes

- prop explosion
- components that try to satisfy every use case
- styling overrides that break accessibility and consistency

### Behavioral ambiguity

The component appears flexible, but users of the API cannot predict focus behavior, loading behavior, or composition contracts correctly.

## Review Questions

- what use case is this API really serving?
- which escape hatch is necessary vs harmful?
- can consumers misuse this component too easily?
- what part of this API will be hardest to deprecate later?
