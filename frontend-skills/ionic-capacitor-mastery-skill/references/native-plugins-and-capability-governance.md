# Native Plugins and Capability Governance

## Principle

Every plugin adds a runtime, permission, and release compatibility surface. Treat plugins like platform dependencies, not small convenience helpers.

## Governance Questions

- who approves plugin adoption
- what permissions does the plugin require
- what fallback exists on unsupported or degraded devices
- how are plugin version upgrades tested

## Adoption Heuristics

### Treat plugins like privileged dependencies

Every plugin should justify its runtime risk, permission model, maintenance burden, and release impact.

### Keep plugin ownership explicit

If a plugin powers a critical workflow, one team should clearly own its lifecycle, upgrades, and incident handling.

### Define graceful degradation

Users should not discover missing capability support only through broken flows or cryptic permission errors.

## Common Failure Modes

- plugin sprawl with unclear ownership
- UI components calling plugins directly without policy boundaries
- permission flows hidden inside convenience wrappers

### Upgrade surprise

Plugin version changes alter behavior or permissions in ways the release process was not built to catch.

## Review Questions

- what risk does this plugin introduce?
- what user experience exists when it fails or is denied?
- how quickly can the team isolate a bad plugin regression?
- what plugin is currently too critical for how weakly it is governed?
