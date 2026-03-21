# Edge vs Server Runtime Tradeoffs for Bun Services

## Purpose

Bun code often looks portable, but deployment constraints differ sharply between edge-style runtimes and full server environments.

## Rules

- decide whether long-lived connections, file access, CPU-heavy work, and local state are valid assumptions
- treat runtime portability as a requirement only when it matters
- do not design to the least common denominator unless the product needs it

## Review Questions

- does this service require long-lived process semantics?
- which dependencies assume full server behavior?
- what runtime assumptions break at the edge?
