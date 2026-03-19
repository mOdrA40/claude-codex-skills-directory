# Security and Outbound HTTP (Bun Services)

This guide focuses on the most common production-grade backend failures in Bun APIs: unsafe input, unbounded outbound calls, SSRF, and secret leakage.

## Threat Model First

Before adding middleware or libraries, answer:

- What inputs are attacker-controlled?
- Which routes trigger side effects or money movement?
- Which endpoints call third-party services or internal-only hosts?
- What secrets or PII might accidentally leak through logs or errors?

## Boundary Defaults

- Validate body, params, query, and headers at the edge.
- Apply request size limits for public endpoints.
- Use deny-by-default CORS and origin allowlists.
- Never pass raw driver or stack errors to clients.
- Redact secrets in logs by default.

## Outbound HTTP Rules

- Every call must have a timeout.
- Retries must be bounded and only applied to safe/idempotent operations.
- Prefer one shared client factory with standard headers, tracing, and timeout policy.
- Do not stack retries in app code, proxy, and worker layers without coordination.

## SSRF Guardrails

Dangerous features include:

- fetch-by-URL endpoints,
- webhook callbacks,
- metadata/image fetchers,
- import/sync jobs that accept external URLs.

Mitigations:

- Allowlist schemes (`http`, `https`) and known hosts.
- Reject loopback, link-local, and private-address targets unless explicitly required.
- Re-validate redirect targets.
- Bound response size and total time.

## Error Handling

Use stable public codes:

- `VALIDATION_ERROR`
- `UNAUTHORIZED`
- `FORBIDDEN`
- `NOT_FOUND`
- `CONFLICT`
- `RATE_LIMITED`
- `DEPENDENCY_UNAVAILABLE`
- `INTERNAL_ERROR`

Log technical cause separately from public payload.

## Secrets Handling

- Keep secrets in env or secret managers.
- Do not print env objects during startup.
- Redact `authorization`, `cookie`, `token`, `password`, and API keys from logs.
- Ensure health endpoints never expose config or dependency credentials.

## Example: Timeout + Classification

```typescript
const controller = new AbortController()
const timer = setTimeout(() => controller.abort(), 2_500)

try {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: controller.signal,
  })

  if (!response.ok) {
    throw new AppError("Dependency unavailable", 503, "DEPENDENCY_UNAVAILABLE")
  }
} finally {
  clearTimeout(timer)
}
```

## Principal Review Lens

Before approving an endpoint or integration, ask:

- Can attacker-controlled input reach file paths, SQL, templates, or outbound URLs?
- What prevents one slow dependency from stalling the whole service?
- What public error do clients see when the dependency fails?
- Which logs/metrics would help on-call distinguish timeout vs auth vs validation failure?
