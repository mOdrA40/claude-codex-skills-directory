# Debugging and Production Incidents in Remix

## Principle

Production Remix incidents usually blend route hierarchy, data flow, enhancement assumptions, and auth behavior. Good debugging starts with route ownership clarity.

## Incident Classes

- loader/action regression
- nested route mismatch
- auth redirect confusion
- route-specific stale or broken UI states

## Review Questions

- is this a route ownership bug, data flow bug, or enhancement bug?
- what release changed route or action behavior?
- how can blast radius be reduced while preserving useful route content?
