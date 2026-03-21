# Routing, Rewrites, and Path Behavior

## Rules

- Routing behavior should be predictable and documented for operators and app teams.
- Rewrites, regex paths, and special annotations increase risk and should justify themselves.
- Header manipulation and forwarded context should remain consistent across services.
- Path matching edge cases become product bugs quickly.

## Common Mistakes

- Rewrites hiding broken app assumptions.
- Conflicting ingress rules across namespaces or teams.
- One-off annotations creating controller behavior surprises.
- Implicit redirect and trailing-slash behavior causing hard-to-debug incidents.

## Principal Review Lens

- Can a reviewer predict exactly where a request goes?
- Which annotation or path rule is most likely to surprise on-call?
- Are we using ingress to compensate for application design issues?
- What routing simplification would cut incident time most?
