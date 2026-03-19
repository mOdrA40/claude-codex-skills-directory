# Log / Trace Correlation

Goal: given a user report, you can find *one* request and follow it through:
- request ID
- trace ID
- user ID (careful; often hash/opaque)

## Minimum fields (services)

- `service`, `env`
- `request_id`
- `trace_id`, `span_id` (if tracing enabled)
- `method`, `path`, `status`, `latency_ms`
- `error_code` (stable) for failures

## Extracting trace IDs from `context.Context`

If you use OpenTelemetry:

```go
sc := trace.SpanContextFromContext(ctx)
if sc.IsValid() {
  traceID := sc.TraceID().String()
  spanID := sc.SpanID().String()
  _ = traceID
  _ = spanID
}
```

## Good vs bad

Bad: `err.Error()` only, no correlation:

```go
log.Printf("failed: %v", err)
```

Good: stable code + trace/request IDs:

```go
logger.Error("request failed", "code", "DB_TIMEOUT", "trace_id", traceID, "request_id", reqID, "err", err)
```

