# Auth, Sessions, and Token Handling in Frontend Applications

## Principle

Frontend auth decisions are security architecture decisions. Token storage, session renewal, and boundary placement affect incident severity directly.

## Design Questions

- which session data belongs in cookies vs browser storage vs memory
- what happens when the session expires mid-flow
- how are auth failures surfaced without causing redirect loops or state corruption
- what browser context is allowed to access authentication state

## Threat-Driven Heuristics

### Prefer boundaries that reduce XSS blast radius

Session convenience is not the only factor. Storage choices should be evaluated against what happens if the browser execution environment is compromised.

### Make session recovery explicit

When sessions expire, refresh, or become partially invalid, users should not bounce through confusing redirect loops that hide the real security state.

### Embedded or multi-origin contexts raise the stakes

If the app is embedded, opened across tabs, or communicates with other browser contexts, auth state assumptions must be extra explicit.

## Common Failure Modes

- treating token storage as a convenience choice instead of a risk tradeoff
- embedding auth assumptions too deeply into route logic
- mixing SSR and client auth resolution with unclear ownership

### Session drift

The browser thinks the user is authenticated while the backend or identity system disagrees, creating ambiguous and potentially unsafe transitions.

## Review Questions

- where can credentials or session state leak?
- what is the blast radius of XSS given current auth storage decisions?
- what recovery behavior exists for partial or expired session state?
- which route or surface becomes most confusing when auth state is only partially valid?
