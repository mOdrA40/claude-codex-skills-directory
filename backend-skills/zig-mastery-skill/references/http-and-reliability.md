# Zig HTTP Services and Reliability

## Goal

This guide covers how to build Zig services that behave predictably under network failures, slow dependencies, retries, and partial outages.

## Service Defaults

Every production service should define:

- request size limits
- header size limits
- read and write deadlines
- upstream timeouts
- graceful shutdown policy
- health and readiness policy
- log schema
- metrics and trace correlation strategy

## Error Mapping

Never leak raw downstream failures directly to clients. Map them into stable categories.

Example categories:

- `400` invalid input
- `401` unauthorized
- `403` forbidden
- `404` not found
- `409` conflict
- `429` rate limited
- `503` dependency unavailable
- `504` upstream timeout
- `500` internal failure

## Bad vs Good: Raw Dependency Error Exposure

```zig
// ❌ BAD: response mirrors internal failure details.
return Response.text(500, err_name);
```

```zig
// ✅ GOOD: client gets stable semantics, operators get details in logs.
logger.err("payment call failed", .{ .error = err_name, .request_id = request_id });
return Response.json(503, .{ .code = "DEPENDENCY_UNAVAILABLE" });
```

## Outbound Calls

For outbound HTTP or RPC:

- set a timeout
- classify safe retries
- add jittered backoff when retrying
- define idempotency expectations
- capture dependency latency separately from end-to-end latency

Blind retries are one of the most common causes of amplified incidents.

## Bad vs Good: Retry Placement

```text
❌ BAD
- transport retries
- client library retries
- upstream queue retries
- no visibility into retry amplification

✅ GOOD
- one owner for retries
- retry budget is explicit
- idempotency is documented
- operators can see retry rate and timeout rate
```

## Backpressure

Backpressure must be designed, not discovered in production.

Possible controls:

- limit request concurrency
- bound worker queues
- reject work early when capacity is exhausted
- degrade optional features before core paths fail
- separate expensive and cheap traffic classes

## Graceful Shutdown

A correct shutdown sequence is:

1. stop accepting new traffic
2. mark readiness unhealthy
3. drain in-flight requests with a deadline
4. stop background workers
5. flush logs/metrics if needed
6. close dependency pools

## Health vs Readiness

### Health

Use health to answer: is the process alive enough to run?

### Readiness

Use readiness to answer: should this instance receive traffic right now?

Readiness should fail during:

- boot before dependencies are ready
- draining shutdown
- unrecoverable dependency loss when the service cannot serve safely

## Incident Questions

- How many requests are timing out?
- Which dependency is dominant in failure contribution?
- Are retries increasing load on a degraded system?
- Is queue growth bounded?
- What is the fastest safe mitigation: shed load, disable a feature, or roll back?

## Reliability Checklist

- Timeouts are explicit.
- Retry ownership is explicit.
- Dependency failures are mapped consistently.
- Readiness behavior matches rollout and drain expectations.
- Expensive request paths are bounded.
- Idempotency is defined for retried side effects.
