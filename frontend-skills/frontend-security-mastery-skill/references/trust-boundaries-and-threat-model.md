# Trust Boundaries and Threat Model for Frontend Systems

## Principle

Frontend security starts with explicit trust boundaries. If the team cannot say what is trusted, what is untrusted, and what can execute in the browser, security controls become checkbox theater.

## Threat-Model Questions

Teams should be able to answer:

- what data is untrusted but renderable
- what code is first-party vs third-party vs partner-controlled
- which browser contexts can exchange data or auth signals
- what happens if an attacker gains script execution in one surface
- what privileged operation is reachable from the frontend at all

## Boundaries to Define

- browser vs server trust
- first-party code vs third-party scripts
- SSR output vs client mutation
- authenticated session vs unauthenticated rendering
- embedded frames and cross-origin communication
- device-local storage vs server-authoritative state

## Common Failure Modes

- trusting sanitized-looking data from unsafe origins
- assuming internal users or partner embeds are low risk
- failing to document which layer owns sanitization or validation

### Trust by familiarity

An internal tool, partner widget, or long-lived script is treated as safe because it is familiar, not because its risk has been modeled correctly.

### Boundary mismatch

One layer assumes another layer validates, sanitizes, or constrains behavior, and the dangerous gap lives in between.

## Review Questions

- what untrusted data can reach rendering?
- which script or origin has more privilege than it should?
- what assumption would break first during a real browser-side security incident?
- what boundary is currently described informally but not enforced clearly?
