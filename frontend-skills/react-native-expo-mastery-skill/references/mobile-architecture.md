# React Native Mobile Architecture

## Principle

A mobile app is not just a smaller web app. It must survive network volatility, backgrounding, device permissions, startup cost, and release channel constraints.

## Boundaries

Separate:

- navigation shell
- feature modules
- server state
- local UI state
- persisted local state
- device capability adapters
- analytics and crash reporting

## Architecture Heuristics

### Keep screens thin

Screens should orchestrate, not own everything. When a screen starts owning:

- permission logic
- local storage access
- upload lifecycle
- sync retry policy
- analytics side effects

it becomes fragile and difficult to reuse or test.

### Separate server state from durable local state

Do not treat cached API responses, offline drafts, and ephemeral UI flags as one state bucket.

- server state changes with network and backend truth
- local durable state survives restart and may need migration
- ephemeral UI state should usually die with the screen

### Device capabilities are high-risk boundaries

Camera, push notifications, file system, biometrics, and location should live behind adapters or hooks with explicit permission and failure behavior.

## Common Failure Modes

### Screen-as-god-object

One screen owns rendering, fetches, mutation retries, permission handling, analytics, and navigation decisions. This creates change amplification and debugging pain.

### Context overuse

Teams push too much state into global context because prop drilling feels inconvenient. The result is broad re-renders and unclear ownership.

### Offline ambiguity

The UI shows stale or local data without telling the user what has actually synced.

## Bad vs Good

```text
❌ BAD
A profile screen owns fetching, optimistic update, image picking, upload retries, and local persistence directly.

✅ GOOD
The screen coordinates feature hooks and use-cases; sync, persistence, and device capabilities have explicit owners.
```

## Review Questions

- what state survives app restart?
- what state is server-owned vs device-owned?
- which screens are coupled only because navigation is unclear?
- where would you look first if sync breaks after app resume?
- which module owns release-sensitive storage migrations?
