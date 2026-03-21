---
name: flutter-principal-engineer
description: |
  Principal/Senior-level Flutter playbook for cross-platform app architecture, rendering performance, state boundaries, offline behavior, platform integration, testing, and production release operations.
  Use when: building or reviewing Flutter applications, choosing app architecture, improving performance, handling navigation/state complexity, integrating native features, or preparing production releases.
---

# Flutter Mastery (Senior → Principal)

## Operate

- Confirm target platforms, rendering constraints, navigation complexity, state-management approach, offline/sync needs, native integration, and release constraints.
- Separate app shell, domain/use-case logic, view state, async data, persistence, and platform channels.
- Optimize for stable rendering, predictable state transitions, and release safety.

## Default Standards

- Keep widget trees declarative and composable.
- Separate ephemeral UI state from domain/app state.
- Avoid framework-wide state magic when smaller boundaries are sufficient.
- Model loading, empty, error, stale, and offline states intentionally.
- Measure frame drops, startup, and rebuild hotspots before optimizing.

## References

- Flutter architecture and state boundaries: [references/architecture-and-state.md](references/architecture-and-state.md)
- Navigation and app flows: [references/navigation-and-app-flows.md](references/navigation-and-app-flows.md)
- Performance and rendering pipeline: [references/performance-and-rendering.md](references/performance-and-rendering.md)
- Offline sync and local persistence: [references/offline-sync-and-persistence.md](references/offline-sync-and-persistence.md)
- Platform channels and native integration: [references/platform-channels-and-native-integration.md](references/platform-channels-and-native-integration.md)
- Testing and release operations: [references/testing-and-release-ops.md](references/testing-and-release-ops.md)
