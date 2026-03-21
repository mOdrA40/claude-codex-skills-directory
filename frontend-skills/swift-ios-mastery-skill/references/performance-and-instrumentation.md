# Performance and Instrumentation on iOS

## Focus Areas

- startup cost
- scroll and animation smoothness
- memory growth
- battery-sensitive background work
- instrumentation with meaningful release visibility

## Investigation Order

1. confirm whether the issue is startup, navigation, scroll, network, or persistence related
2. identify the dominant screen, workflow, or device class
3. correlate the issue with release version and feature exposure
4. determine whether the bottleneck is CPU, memory, IO, animation, or background behavior

## Common Failure Modes

### Slow startup from doing too much too early

Apps often initialize analytics, storage, remote config, session restore, and multiple fetches before the user sees meaningful UI.

### Smooth on simulator, poor on real devices

Teams optimize on comfortable hardware and miss memory or animation issues on lower-end devices.

### Observability without actionability

Crash and metrics tools exist, but operators cannot answer which release, route, or feature introduced the regression.

## Review Questions

- what work can move later without harming correctness?
- which flows are most sensitive to scroll or animation jank?
- can you correlate user pain to app version, device segment, and screen?
- which background tasks are spending battery without strong product value?
