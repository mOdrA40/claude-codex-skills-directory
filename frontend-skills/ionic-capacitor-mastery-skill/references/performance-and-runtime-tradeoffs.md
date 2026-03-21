# Performance and Runtime Tradeoffs in Ionic + Capacitor

## Principle

Hybrid performance work must consider rendering cost, web asset delivery, bridge overhead, and device-specific runtime variability together.

## Runtime Model

Hybrid apps combine several performance domains:

- web rendering and asset delivery
- bridge invocation cost
- device and OS variability
- local persistence and sync behavior
- shell bootstrap and resume behavior

A fast webview alone does not guarantee a responsive hybrid product.

## Common Failure Modes

- over-optimizing bundle size while bridge or device APIs dominate pain
- assuming one device profile represents the whole user base
- slow startup caused by too much bootstrap work in the web shell

### Bridge-heavy interaction loops

Frequent native calls inside interactive flows make the app feel inconsistent even when raw web rendering seems acceptable.

### Resume-time chaos

An app feels fine on cold start tests but behaves poorly when resuming, restoring session state, or reconnecting after background time.

## Investigation Heuristics

### Separate web pain from hybrid pain

Ask whether the user is suffering from:

- rendering and layout work
- network and asset loading
- bridge latency
- permission and plugin lifecycle behavior
- resume or restore complexity

## Review Questions

- is the bottleneck rendering, asset delivery, bridge calls, or lifecycle work?
- what device classes are most affected?
- what work can be deferred after first meaningful interaction?
- what app behavior feels mobile-slow even though the web surface alone looks acceptable?
