# Performance and Rendering in React Native

## Common Bottlenecks

- expensive lists without virtualization discipline
- unnecessary re-renders from unstable props
- oversized app startup work
- image and media handling overhead
- JS thread contention from too much orchestration

## Investigation Order

1. identify whether the pain is startup, navigation transition, list rendering, media handling, or sync-triggered UI churn
2. determine whether the bottleneck is on the JS thread, native side, network, or storage path
3. compare behavior across lower-end and higher-end devices
4. correlate regressions with release version and feature exposure

## Common Failure Modes

### One heavy screen poisons the whole app

Expensive rendering, large images, or too much orchestration on a frequently visited screen can make the app feel globally slow.

### Startup does too much

Session restore, analytics, permissions, remote config, and initial fetches all pile up before users reach meaningful UI.

### Sync and render fight each other

Frequent state updates, cache writes, or optimistic flows produce churn that users feel as lag or stutter.

## Review Questions

- which route or feature dominates perceived slowness?
- what work blocks first meaningful interaction?
- is the bottleneck rendering, network, storage, or thread contention?

## Review Questions

- is the bottleneck startup, navigation, list rendering, or device capability cost?
- what work happens before first meaningful paint?
- can one noisy screen degrade the whole app experience?
