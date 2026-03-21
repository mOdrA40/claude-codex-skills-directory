# Module Design and Composition (Terraform)

## Rules

- Modules should express stable platform intent, not mirror provider APIs one-to-one blindly.
- Reuse should reduce cognitive load, not create nested abstraction mazes.
- Input/output design should reveal ownership, dependencies, and safe usage boundaries.
- Keep modules reviewable and easy to test mentally.

## Design Heuristics

- Separate foundational networking, identity, data, and app platform concerns when blast radius differs.
- Prefer explicit interfaces over magic defaults that hide infrastructure consequences.
- Allow extension only where real reuse pressure exists.
- Document invariants through code structure and naming discipline.

## Principal Review Lens

- Does this module reduce real duplication or merely hide it?
- What change inside this module has the widest hidden blast radius?
- Can another team understand the interface without reading provider internals?
- Which module boundary should be split because ownership is already divergent?
