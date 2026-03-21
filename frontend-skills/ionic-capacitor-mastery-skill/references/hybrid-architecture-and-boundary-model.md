# Hybrid Architecture and Boundary Model

## Principle

Hybrid apps succeed when the team understands which concerns belong to web UI architecture and which belong to native bridge boundaries. Confusion here creates slow, fragile, hard-to-debug apps.

## Separate Clearly

- web rendering and routing
- native plugin access
- local persistence
- auth/session handling
- offline sync
- release and upgrade behavior

## Architecture Heuristics

### Keep native bridge calls away from view code

If components call native capabilities directly, debugging and permission behavior become hard to reason about across platforms.

### Treat hybrid lifecycle as first-class

Resume behavior, deep links, offline recovery, and session restoration must be modeled as part of architecture, not left to framework defaults.

### Distinguish browser-like behavior from mobile-operational behavior

The app may render with web technologies, but release, permissions, storage, and lifecycle pressures behave more like mobile than classic web.

## Common Failure Modes

- treating every mobile problem as a web problem
- scattering bridge calls through UI components
- assuming plugin behavior is uniform across devices and OS versions

### Shell confusion

The team cannot explain what part of the experience is owned by the web shell, what part by native integration, and what part by backend or sync policy.

## Review Questions

- what boundary owns device capability access?
- what part of the app is truly cross-platform vs only seemingly so?
- where would incident ownership be unclear today?
- which lifecycle or bridge assumption is most likely to break in production first?
