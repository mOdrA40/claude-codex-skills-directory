# Backend Incident Postmortem Patterns for Bun Services

## Common Patterns

- timeout storms amplified dependency pain
- queue lag hid correctness issues
- webhook retries duplicated side effects
- rollout compatibility gap broke mixed-version traffic
- one tenant consumed shared worker or dependency budget

## Good Follow-Up Questions

- what signal should have existed earlier?
- what containment could be automated?
- what change policy should tighten next time?
- what optional work should degrade sooner?
