# Performance and Rendering Discipline

## Focus Areas

- startup cost
- recomposition discipline
- list performance
- memory and background resource usage

## Investigation Heuristics

### Startup pain

Look for excessive bootstrap work, storage restoration overhead, eager network calls, and feature initialization that can move later.

### Recomposition pain

Look for state ownership that is too broad, unnecessary recomposition cascades, and UI models that change more often than necessary.

### List and scroll pain

Look for image cost, unstable item state, and UI models that force expensive updates during interaction.

## Common Failure Modes

### Broad state invalidation

One state change triggers too much UI churn because ownership boundaries are too coarse.

### Good on emulator, weak on real devices

Teams validate on ideal hardware and miss actual budget pressure on target devices.

### Background work stealing UX budget

Sync, persistence, or analytics work competes with user-visible performance without enough operational value.

## Review Questions

- what user-visible pain is the optimization meant to fix?
- which state changes trigger the widest recomposition impact?
- what work can move later without harming correctness?
