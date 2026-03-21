# Rendering and Update Cost

## Principle

Rendering performance is about the scope and frequency of work, not just whether a single component looks expensive in isolation.

## Common Failure Modes

- over-shared state causing broad re-renders
- list or table updates affecting far more UI than intended
- derived calculations repeated without meaningful benefit

## Investigation Heuristics

### Find the widest blast radius first

The most expensive update is often not the most obvious component. It is the state change that touches the most surfaces.

### Distinguish frequency from cost

Some updates are cheap but happen constantly. Others are expensive but rare. Both matter differently.

### Tie rendering work to user value

If the UI updates more often than the user can perceive or benefit from, the architecture is doing unnecessary work.

## Review Questions

- what state change causes the widest update blast radius?
- what part of the UI updates more than the user value justifies?
- what ownership boundary would reduce work the most?
- which rendering cost is currently hidden behind “it feels okay on my machine”?
