# Frameworks (Gin / Fiber / Beego) — Best Practices

Default rule: framework is just transport. Keep domain/use-cases independent from Gin/Fiber/Beego so you can test and migrate.

## Cross-framework rules

- Centralize error handling (one place decides status code + error envelope).
- Put request limits at the edge (body size, timeouts, rate limits).
- Use panic recovery middleware in production.
- Validate input at the boundary; don’t let invalid data reach domain logic.
- Correlation IDs: request ID + trace ID + structured logs.

For status-code conventions and error shape, see `http-api.md`.

## Gin (`gin-gonic/gin`)

### Routing + middleware

- `gin.Default()` includes Logger + Recovery.
- For custom stack, use `gin.New()` then `r.Use(gin.Logger())` and `r.Use(gin.Recovery())` or `gin.CustomRecovery(...)`.

### Binding/validation

- Prefer `c.ShouldBind...` (e.g. `ShouldBindJSON`) and return `400` on bind/validation errors.
- Use struct tags `binding:"required"` etc for request DTOs.
- If you want strict JSON (no unknown fields), add a custom binder/decoder strategy; otherwise standard binding won’t reject extra fields by default.

### Timeouts

Gin runs on `net/http`; configure timeouts on the underlying `http.Server` (Read/Write/ReadHeader/Idle) and use `http.NewRequestWithContext` for outbound calls.

## Fiber (`gofiber`)

Fiber is optimized and Express-like; be careful with assumptions from `net/http`.

### Central error handler + recovery

- Add `recover` middleware to prevent crashes from panics.
- Standardize error responses by configuring the app’s error handler (so you don’t duplicate JSON error formatting per handler).

### Timeouts

- Use Fiber timeout middleware per-route when you need deterministic time caps.
- Ensure downstream calls use `ctx.Context()`-derived cancellation/deadlines when supported by the dependency.

### Request body parsing

- Don’t store `BodyRaw()` results beyond the handler lifetime; treat it as request-scoped.
- Enforce body size limits at the edge (proxy) and/or via Fiber config/middleware.

## Beego (`beego/beego`)

Beego has batteries-included patterns; keep them isolated from core logic.

### Validation

- Use Beego validation tags (`valid:"Required;Range(...)"`) and run validation at the boundary.
- Custom validators can be registered for domain-specific rules; keep those rules close to DTOs, not to deep domain internals.

### Sessions/logging

- If you use sessions, treat session data as untrusted input (it can be tampered depending on backing store/config).
- Configure logging outputs/levels explicitly and keep logs structured if possible.

## Choosing between them (pragmatic)

- Prefer `net/http` compatibility + ecosystem instrumentation: Gin (or stdlib + chi).
- Prefer Fiber when you need its ergonomics/perf model and accept its differences from `net/http`.
- Prefer Beego when you’re in an existing Beego codebase; avoid starting new projects with heavy framework coupling unless you need its integrated features.
