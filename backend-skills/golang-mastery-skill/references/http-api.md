# HTTP API (Response Codes + Error Shape)

## Goals

- Clients can reliably branch on `status` and a stable `code`.
- Humans can debug via `message`, `details`, and correlated IDs.
- Errors are consistent across handlers and frameworks.

## Recommended Error Envelope (JSON)

```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "user not found",
    "details": {
      "user_id": "123"
    }
  },
  "request_id": "…",
  "trace_id": "…"
}
```

Rules:
- `code`: stable, UPPER_SNAKE, never changes once public.
- `message`: safe for clients (no secrets), short and actionable.
- `details`: optional; avoid PII; for validation errors include field-level info.
- Include `request_id` and/or `trace_id` for correlation (even if you don’t expose internals).

## Status Code Mapping (sane defaults)

### 2xx
- `200 OK`: successful GET/PUT/PATCH; response body present.
- `201 Created`: resource created; include `Location` header when you have canonical URL.
- `202 Accepted`: async processing started.
- `204 No Content`: successful delete or update with no body.

### 3xx
- Avoid in APIs unless you control the client. If used, re-validate redirects to prevent SSRF/open redirect.

### 4xx (client errors)
- `400 Bad Request`: malformed payload, invalid JSON, failed schema validation, unknown fields (if strict).
- `401 Unauthorized`: missing/invalid authentication.
- `403 Forbidden`: authenticated but not allowed.
- `404 Not Found`: missing resource (or hide existence by using 404 for authz-sensitive resources).
- `409 Conflict`: unique constraint violations, version conflicts (optimistic locking).
- `412 Precondition Failed`: `If-Match`/ETag preconditions fail.
- `415 Unsupported Media Type`: wrong `Content-Type`.
- `422 Unprocessable Entity`: syntactically valid but semantically invalid (optional; choose either 400 or 422 and standardize).
- `429 Too Many Requests`: rate limit exceeded; include `Retry-After` if meaningful.

### 5xx (server errors)
- `500 Internal Server Error`: unexpected error.
- `502 Bad Gateway`: upstream invalid response.
- `503 Service Unavailable`: dependency down / overload; prefer with load shedding.
- `504 Gateway Timeout`: upstream timed out (edge gateway usually sets this).

## Good vs bad (handler responses)

Bad: inconsistent shapes and status codes:

```go
if err != nil {
  c.JSON(200, gin.H{"error": err.Error()})
  return
}
```

Good: stable status + stable error envelope:

```go
if err != nil {
  // map err -> status + code, then respond once
  c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"code": "INVALID_INPUT", "message": "invalid input"}})
  return
}
```

## Headers & Limits (security + reliability)

- Set request size limits; reject early.
- Always set timeouts (server + client + per-request context).
- Consider these headers (depends on environment):
  - `Content-Type: application/json`
  - `Cache-Control` for cacheable endpoints
  - `ETag` + `If-Match` for optimistic concurrency on updates

## Pagination (avoid OFFSET for large data)

- Prefer keyset pagination:
  - request: `?limit=50&cursor=…`
  - response: `next_cursor`
- Always cap `limit` and validate it.

## Idempotency (payments/side effects)

- For “create with side effect”: accept `Idempotency-Key` and enforce uniqueness server-side.
- Return the same result for the same key (within a retention window).
