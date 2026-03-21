---
name: swift-ios-principal-engineer
description: |
  Principal/Senior-level Swift iOS playbook for app architecture, SwiftUI/UIKit boundaries, async state, performance, offline behavior, native platform capabilities, testing, and production delivery.
  Use when: building or reviewing iOS applications, choosing SwiftUI/UIKit boundaries, improving app architecture, handling data flow and offline state, optimizing performance, or preparing App Store releases.
---

# Swift iOS Mastery (Senior → Principal)

## Operate

- Confirm iOS deployment target, Swift/SwiftUI adoption level, UIKit interop needs, async data flow, offline requirements, app lifecycle constraints, and release requirements.
- Separate app shell, navigation, view state, domain logic, persistence, platform capabilities, and analytics/crash reporting.
- Optimize for maintainability, performance, and predictable platform behavior.

## Default Standards

- prefer SwiftUI by default but keep UIKit interop explicit
- keep view logic thin and state ownership clear
- model loading, empty, error, offline, and degraded states intentionally
- isolate platform services such as notifications, camera, biometrics, and background tasks
- measure startup, scrolling, memory, and battery-sensitive behavior before optimizing

## References

- iOS architecture and state ownership: [references/ios-architecture-and-state.md](references/ios-architecture-and-state.md)
- SwiftUI and UIKit boundaries: [references/swiftui-uikit-boundaries.md](references/swiftui-uikit-boundaries.md)
- Navigation and app lifecycle: [references/navigation-and-lifecycle.md](references/navigation-and-lifecycle.md)
- Offline persistence and sync: [references/offline-persistence-and-sync.md](references/offline-persistence-and-sync.md)
- Failure states and resilience: [references/failure-states-and-resilience.md](references/failure-states-and-resilience.md)
- Performance and instrumentation: [references/performance-and-instrumentation.md](references/performance-and-instrumentation.md)
- Release operations and store readiness: [references/release-ops-and-store-readiness.md](references/release-ops-and-store-readiness.md)
