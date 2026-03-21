---
name: kotlin-kmp-jetpack-compose-principal-engineer
description: |
  Principal/Senior-level Kotlin, Kotlin Multiplatform, and Jetpack Compose playbook for Android/mobile architecture, shared business logic, UI state, offline sync, performance, release operations, and platform boundary design.
  Use when: building or reviewing Android or Kotlin Multiplatform applications, designing shared modules, structuring Compose UI, improving app performance, or preparing production releases across platforms.
---

# Kotlin / KMP / Jetpack Compose Mastery (Senior → Principal)

## Operate

- Confirm Android/KMP targets, shared-module goals, platform ownership boundaries, offline requirements, UI complexity, native integrations, and release constraints.
- Separate UI, state holders, domain logic, persistence, sync, and shared-vs-platform-specific boundaries.
- Optimize for correctness, maintainability, and release-safe modularity rather than shared-code vanity.

## Default Standards

- use shared code only when ownership and lifecycle justify it
- keep Compose state and domain state distinct
- model loading, empty, error, offline, and retry behavior clearly
- isolate platform-specific capabilities behind explicit adapters
- measure rendering, startup, and synchronization cost before optimizing

## References

- KMP architecture and shared boundaries: [references/kmp-architecture-and-shared-boundaries.md](references/kmp-architecture-and-shared-boundaries.md)
- Compose state and UI modeling: [references/compose-state-and-ui-modeling.md](references/compose-state-and-ui-modeling.md)
- Navigation and app flows: [references/navigation-and-app-flows.md](references/navigation-and-app-flows.md)
- Offline sync and local persistence: [references/offline-sync-and-local-persistence.md](references/offline-sync-and-local-persistence.md)
- Failure states and resilience: [references/failure-states-and-resilience.md](references/failure-states-and-resilience.md)
- Performance and rendering discipline: [references/performance-and-rendering-discipline.md](references/performance-and-rendering-discipline.md)
- Release operations and compatibility: [references/release-operations-and-compatibility.md](references/release-operations-and-compatibility.md)
