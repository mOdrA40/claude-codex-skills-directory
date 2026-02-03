# Security (Senior+ Checklist)

This is intentionally “boring security”: practical, high-signal controls that prevent the most common real incidents.

## Threat Model (before code)

- Assets: credentials, tokens, PII, money-moving operations, admin endpoints, internal network access, build pipeline.
- Trust boundaries: internet → edge → service → DB/cache/queue → third parties.
- Attacker capabilities: untrusted input, replay, credential stuffing, SSRF, data exfiltration, DoS, supply-chain.
- Failure modes: what happens when auth fails, dependency is down, timeouts hit, partial writes occur.

## Secure Defaults (Go-specific)

- Put hard limits at the boundary:
  - request body: `http.MaxBytesReader`
  - JSON: `Decoder.DisallowUnknownFields()` for strict APIs
  - timeouts everywhere (`context.WithTimeout`, client/server timeouts)
- Deny-by-default, allowlist explicitly (CORS, file paths, outbound hosts, redirect targets).
- Never log secrets/PII; design log fields as “safe by default”.
- Treat any string from outside as hostile (headers, URLs, file names, env, DB values from user-controlled rows).

See also:
- Outbound HTTP hardening: `outbound-http.md`
- Auth hardening: `auth.md`

## Common “gotchas” that cause real vulns

### HTTP client defaults (silent DoS)

Bad: no timeout, can hang forever:

```go
resp, err := http.Get(url)
```

Good: client with timeouts + bounded reads:

```go
client := &http.Client{Timeout: 10 * time.Second}
req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
resp, err := client.Do(req)
```

### SSRF (server-side request forgery)

Patterns that get people owned:
- “fetch URL” endpoints
- webhooks that call back arbitrary URLs
- image/metadata fetchers

Mitigations:
- Allowlist schemes (`http/https` only) and hosts (exact hostnames or suffix allowlists).
- Resolve DNS and block private/loopback/link-local ranges (`127.0.0.0/8`, `10.0.0.0/8`, `169.254.0.0/16`, `::1`, etc.).
- Disable automatic redirects or re-validate on every redirect hop.
- Use a dedicated `http.Client` with:
  - strict timeouts
  - capped response sizes (`io.LimitReader`)
  - no proxy from env unless explicitly required

### Path traversal & “zip slip”

If you write/read files based on user input:
- Clean and validate: `filepath.Clean`, reject absolute paths, reject `..` escapes.
- Enforce “must stay under base dir” with `filepath.Rel` check.
- For archives (zip/tar): validate each entry path before extracting.

### JSON parsing pitfalls (API confusion)

Bad: accept unknown fields and silently ignore typos:

```go
_ = json.NewDecoder(r.Body).Decode(&req)
```

Good (strict APIs): reject unknown fields:

```go
d := json.NewDecoder(r.Body)
d.DisallowUnknownFields()
if err := d.Decode(&req); err != nil { ... }
```

### SQL injection

- Always parameterize queries. Never use `fmt.Sprintf` to build SQL with user input.
- If you must build dynamic SQL:
  - allowlist column names and sort directions
  - keep values parameterized

### JWT / session pitfalls

- Validate `iss`, `aud`, `exp`, `nbf`, `iat` and clock skew.
- Don’t accept arbitrary algorithms; pin acceptable algs and key types.
- Plan key rotation (kid, JWKS caching rules, emergency revoke).
- Prefer short-lived access tokens + refresh token rotation for user sessions.

### Crypto footguns

- Use `crypto/rand` for tokens/keys.
- Password storage: use a KDF (`bcrypt`/`argon2id`); never plain hashes.
- Use constant-time comparisons for secrets (`crypto/subtle`).
- Don’t invent crypto protocols; use established libraries and formats.

## Supply Chain / Build Hardening

- Always: `go mod verify` in CI.
- Prefer reproducible builds:
  - `go build -trimpath`
  - decide if you want VCS stamping (Go’s default `-buildvcs`) and make it consistent across CI.
- Lock down private module access (`GOPRIVATE`, `GONOSUMDB`) and avoid leaking module paths in logs.
- Run vulnerability scanning as a gate: `govulncheck ./...` (and review results, don’t auto-ignore).

## Minimum “Prod Gate” (security)

- `go test ./...` (+ `-race` where relevant)
- `golangci-lint run` (tuned to reduce noise)
- `govulncheck ./...`
- Secrets scanning in repo/CI (choose a tool; enforce blocking rules for real secrets)
