# Backend Incident Postmortem Patterns for Node.js Services

## Purpose

Postmortems should change system behavior, not only document pain.

## Common Patterns

- retries amplified dependency outage
- queue backlog hid user-visible failure until too late
- deploy compatibility gap caused mixed-version breakage
- event loop saturation was mistaken for generic CPU pressure
- one tenant consumed shared safety margin

## Good Follow-Up Questions

- what signal was missing or too noisy?
- what containment action should become automatic next time?
- what design choice amplified blast radius?
- what release guard should be added?
