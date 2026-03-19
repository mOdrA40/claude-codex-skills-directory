# Auth (Sessions, JWT, OAuth2) — Practical Hardening

## First principle

Auth bugs are usually *state* bugs:
- session lifetime and rotation
- token revocation/rotation
- replay resistance (idempotency keys, nonce)
- unsafe “remember me” implementations

## Passwords (if you store them)

- Hash with a KDF (bcrypt/argon2id/scrypt). Never store raw passwords or plain hashes.
- Always compare with constant-time functions (bcrypt compare does this).
- Add rate limits and lockouts (credential stuffing is the default internet background noise).

Good:

```go
hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
if err != nil { return err }
if err := bcrypt.CompareHashAndPassword(hash, []byte(password)); err != nil {
  return ErrInvalidCredentials
}
```

## Sessions (cookie-based)

Defaults:
- Cookie flags: `HttpOnly`, `Secure`, `SameSite=Lax` (or `Strict` if it doesn’t break flows).
- Rotate session IDs on privilege change (login, role elevation).
- Keep TTL short; implement sliding sessions only if you really need it.
- Invalidate server-side on logout (token blacklist / session store delete).

## JWT (when you must)

Use JWT for stateless *access tokens*, not for long-lived sessions unless you have strong revocation story.

Hard rules:
- Pin allowed algorithms (don’t accept `none`; don’t accept unexpected algs).
- Validate `iss`, `aud`, `exp`, `nbf` consistently; allow small clock skew.
- Keep access token TTL short (minutes), refresh tokens longer with rotation.
- Plan key rotation (`kid`), JWKS caching rules, and emergency revoke.
- Never put secrets/PII you can’t leak into JWT claims.

Bad:

```go
// Accepts any token header alg + missing claim validation.
token, _ := jwt.Parse(tokenString, keyFunc)
_ = token
```

Good (shape):

```go
// 1) restrict algs
// 2) validate registered claims (iss/aud/exp/nbf)
// 3) treat any parse/validation error as auth failure
```

## OAuth2 / OIDC

Rules of thumb:
- Prefer OIDC over “raw OAuth2” when you need identity.
- Always validate issuer and audience; don’t accept tokens from “any issuer”.
- Cache JWKS with sane TTL; handle key rotation gracefully.
- Treat upstream outages as an availability risk: add timeouts + circuit breakers.

## Authorization (authz)

- Authn answers “who are you?”, authz answers “are you allowed?”
- Put authz decisions close to use-cases (not only in handlers), so batch jobs and internal callers can’t bypass it.
- Hide resource existence when necessary (return 404 for forbidden resources to prevent enumeration).

