# Libraries (Conservative Shortlist)

Default rule: start with stdlib; add dependencies only when they pay for themselves.

## Config (env → struct)

Prefer a maintained env decoder library when configs grow beyond a few vars.

### `sethvargo/go-envconfig` (example)

```go
type Config struct {
  Port string `env:"PORT, default=5555"`
  DSN  string `env:"DATABASE_URL, required"`
}
```

Useful features:

- `required` fields
- `default=` values
- nested struct prefixes (e.g. `prefix=CACHE_`)
- `time.Duration` decoding

## HTTP routing

- Prefer stdlib `net/http` + minimal router.
- Common choice: `chi` (small surface, idiomatic middleware style).

## Logging

- Prefer stdlib `log/slog` for structured logs when it fits.
- If you need a mature, high-performance logger: `zap` or `zerolog` (choose one and standardize).

## Database

- Prefer `database/sql` and keep query boundaries explicit.
- For Postgres, `pgx` is a common driver/pool choice.

## Testing

- Prefer stdlib `testing` + table tests.
- Add assertion libs only if it improves readability and consistency (and the team agrees).

## Observability

- Tracing/metrics: OpenTelemetry is the default ecosystem choice; standardize attribute names and sampling.

