# Embedded Apps, Iframes, and postMessage Risks

## Principle

Embedded apps create security problems because they blur origin, trust, and interaction boundaries. Convenience integrations easily become privilege escalation paths.

## High-Risk Areas

- postMessage origin validation
- message schema validation
- iframe embedding permissions
- parent-child auth assumptions
- partner or internal embeds treated as inherently safe

## Design Heuristics

### Validate both origin and message shape

Checking origin alone is not enough if the message contract is broad or ambiguous.

### Keep privileged actions rare and explicit

If an embedded context can trigger sensitive behavior, the contract should be narrow, typed, and easy to audit.

### Treat partner and internal embeds as real trust boundaries

“Internal” does not mean harmless. Browser context and origin rules still matter.

## Common Failure Modes

- wildcard target origins
- message handlers accepting structurally vague payloads
- auth or session assumptions leaking across embed boundaries

### Message protocol drift

The application evolves, but message schemas are never versioned or governed, creating unsafe compatibility behavior.

## Review Questions

- which origins are allowed to talk to this app?
- how are incoming messages validated?
- what privileged action could be triggered by a bad embed or spoofed message?
- what message path would be hardest to secure retroactively after broad adoption?
