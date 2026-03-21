---
name: react-native-expo-principal-engineer
description: |
  Principal/Senior-level React Native + Expo playbook for mobile architecture, navigation, async state, offline behavior, performance, reliability, native boundaries, and production delivery.
  Use when: building or reviewing React Native/Expo applications, designing mobile architecture, handling navigation and server state, improving startup/render performance, hardening offline and sync flows, preparing app store releases, or debugging production mobile issues.
---

# React Native + Expo Mastery (Senior → Principal)

## Operate

- Start by confirming: Expo workflow, RN version, platform targets, offline requirements, navigation complexity, device/API integrations, performance budget, release constraints, and definition of done.
- Separate boundaries clearly: app shell, navigation, feature state, server state, device capabilities, storage/sync, and analytics/observability.
- Optimize for operability, not just developer speed: release channels, crash visibility, graceful degradation, retry policy, and startup cost all matter.
- Prefer boring mobile correctness over clever abstraction: predictable screen state, explicit sync boundaries, and platform-aware UX beat architectural vanity.

> The goal is not just a working mobile UI. The goal is an app that stays predictable across devices, networks, store releases, and real user behavior.

## Default Standards

- Keep navigation boundaries explicit and route ownership clear.
- Treat server state and local UI state differently.
- Model offline, retry, loading, empty, and stale states intentionally.
- Keep device capability access behind adapters or hooks.
- Avoid hidden global state and overpowered context trees.
- Measure startup, list rendering, and memory hotspots before optimizing.
- Keep release and OTA safety rules explicit.

## “Bad vs Good” (common production pitfalls)

```tsx
// ❌ BAD: fetch directly in screen effect and model no retry/offline state.
useEffect(() => {
  fetch('/api/feed').then(setFeed)
}, [])

// ✅ GOOD: use explicit server-state management and recovery behavior.
const feedQuery = useQuery(feedQueries.list())
```

```tsx
// ❌ BAD: screen owns device permission + network call + storage sync inline.
async function onOpenCamera() {
  const granted = await requestPermission()
  if (granted) {
    const photo = await launchCameraAsync()
    await uploadPhoto(photo)
  }
}

// ✅ GOOD: capability, upload, and sync logic are separated.
await captureAndUploadAvatar.execute()
```

## Validation Commands

- Run `npm test` or repository test command.
- Run `npx tsc --noEmit`.
- Run `expo doctor` when available.
- Run lint and format commands configured by the project.
- Validate release build and bundle size before shipping.

## References

- Mobile architecture and boundaries: [references/mobile-architecture.md](references/mobile-architecture.md)
- Navigation and screen flows: [references/navigation-and-screen-flows.md](references/navigation-and-screen-flows.md)
- Server state, cache, and sync: [references/server-state-and-sync.md](references/server-state-and-sync.md)
- Offline-first and local persistence: [references/offline-first-and-persistence.md](references/offline-first-and-persistence.md)
- Performance and rendering: [references/performance-and-rendering.md](references/performance-and-rendering.md)
- Native capabilities and boundaries: [references/native-capabilities-and-boundaries.md](references/native-capabilities-and-boundaries.md)
- Release, OTA, and incident response: [references/release-ota-and-incidents.md](references/release-ota-and-incidents.md)
