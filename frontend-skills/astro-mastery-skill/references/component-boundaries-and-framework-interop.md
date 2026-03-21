# Component Boundaries and Framework Interop

## Principle

Framework interop is useful in Astro, but it should be governed. Mixing islands from multiple ecosystems carelessly creates ownership and debugging confusion.

## Common Failure Modes

- importing multiple framework islands without clear ownership
- duplicated interaction logic across islands
- teams unable to explain which framework owns which behavior on a page

## Review Questions

- what framework is responsible for this interactive surface?
- is interop reducing delivery risk or increasing it?
- what page would become hardest to debug because of mixed ownership?
