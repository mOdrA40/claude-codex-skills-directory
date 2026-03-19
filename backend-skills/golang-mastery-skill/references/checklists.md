# Go Senior Checklists

## PR Review (General)

- API: small surface area, clear naming, minimal exports, docs for exported identifiers.
- Errors: no swallowed errors; wrapped with `%w` when adding context; no double-logging.
- Context: `ctx` passed and honored; cancellations/timeouts applied at boundaries.
- Testing: new behavior covered; flaky timing avoided; table tests used where helpful; tests deterministic.
- Dependencies: new deps justified; module versions sane; no unnecessary transitive bloat.
- Observability: logs are structured and useful; no secrets; metrics/tracing hooks where required.
- Config: defaults safe; validation present; env flags documented.
- Security: untrusted input bounded; authn/authz enforced; no secrets in logs; outbound calls hardened (SSRF); file ops hardened (path traversal).
- Data/DB: queries parameterized; rows closed; transactions minimal; pagination safe; indexes considered for hot paths.
- Errors: `errors.Is/As` works (wrapping via `%w`); public error codes stable; mapping centralized.

## Concurrency

- Goroutines: lifecycle owned; no leaks; cancellation path exists; `WaitGroup` used correctly.
- Channels: closed by the sender; no sends on closed channels; select includes `ctx.Done()` where needed.
- Locks: minimal critical sections; no lock ordering hazards; no copying mutex values.
- Time: avoid `time.Sleep` in production paths; avoid magic durations in tests.

## HTTP/API

- Timeouts: server and client timeouts set; request bodies bounded via `MaxBytesReader` where appropriate.
- Status codes: consistent mapping; error response shape stable; idempotency considered.
- Retries: only on safe operations; jittered backoff; time-bounded; cancellation-aware.
- Security: authz enforced; inputs validated; CORS only when required; rate limits considered.
- Framework glue: transport layer kept thin; domain/use-case code not tied to Gin/Fiber/Beego types.
- Side effects: idempotency keys for create/payment flows; outbox considered for event publishing.

## Storage

- Transactions: scope minimal; errors handled; retries for transient failures (if DB supports).
- Queries: context used; rows closed; scan errors checked; N+1 avoided for hot paths.
- Migrations: schema changes safe; backwards compatible when required by deployment strategy.
