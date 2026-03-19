# Auth and Session Security (Bun Services)

Authentication failures are rarely caused by missing libraries. They are usually caused by weak session design, poor secret handling, or leaky boundaries.

## Default Position

- Prefer server-side sessions for browser apps unless stateless JWT is clearly justified.
- Use short-lived access tokens and rotation for refresh credentials when token-based auth is required.
- Keep authn and authz separate in design and code.

## Session Guardrails

- Session cookies should be `HttpOnly`, `Secure`, and have appropriate `SameSite` policy.
- Rotation matters after login, privilege elevation, and refresh.
- Logout should invalidate server-side session state, not only delete a cookie.
- Session storage should have TTL and revocation strategy.

## JWT Guardrails

- Pin acceptable algorithms.
- Validate issuer, audience, expiry, not-before, and subject semantics.
- Plan key rotation deliberately.
- Do not stuff sensitive or mutable authorization state into tokens.

## Authorization

- Resolve actor and tenant context explicitly.
- Enforce least privilege.
- Never trust front-end role hints without server verification.
- Log auth failures without leaking sensitive detail.

## Password Handling

- Prefer `Bun.password` with strong algorithm settings.
- Never log passwords or password-derived data.
- Add rate limiting and lockout/slowdown for login and reset flows.
- Treat password reset tokens like secrets.

## Principal Review Lens

Ask:

- What happens if a session token is replayed?
- How is session revocation enforced across instances?
- What is the blast radius of one stolen secret?
- Can auth failures be distinguished from dependency failures during incident response?
