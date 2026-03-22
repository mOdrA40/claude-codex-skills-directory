# Backend Incident Postmortem Patterns for Go Services

## Common Patterns

- retries amplified downstream outages
- goroutine or queue growth hid the real bottleneck
- rollout compatibility gap broke mixed-version traffic
- one tenant or workload shape consumed shared budget
- lock or pool contention was misdiagnosed as generic slowness

## Good Follow-Up Questions

- what signal was missing?
- what containment should become standard next time?
- what design decision amplified blast radius?
- what release guard should be added?
