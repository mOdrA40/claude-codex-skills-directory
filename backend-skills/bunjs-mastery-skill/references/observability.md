# Observability (Bun Services)

Observability is not “extra logging”. It is the minimum information needed to answer: what is failing, where, and why.

## Minimum Signals

- Structured logs.
- Request rate, errors, and duration.
- Dependency latency and failure rates.
- Health and readiness endpoints.
- Request correlation IDs across logs.

## Logging Defaults

Use JSON logs in production and include:

- `service`
- `env`
- `version`
- `requestId`
- `route`
- `method`
- `statusCode`
- `latencyMs`
- `tenantId` when relevant

Do not log secrets, raw tokens, cookies, or full sensitive payloads.

## Metrics Defaults

Track at minimum:

- HTTP request count
- HTTP error count
- HTTP duration histogram
- DB query duration
- Redis latency/error count
- outbound HTTP duration/error count
- queue depth or background-job backlog where relevant

## Request Correlation

- Accept inbound `x-request-id` if trusted, or generate one.
- Return the request ID to clients when appropriate.
- Attach it to all logs emitted during the request lifecycle.

## Health Endpoints

Use separate endpoints:

- `/health` for liveness
- `/ready` for readiness

`/ready` should fail when critical dependencies are degraded enough that the instance should stop receiving traffic.

## Principal Review Lens

Ask:

- Can on-call isolate one broken request quickly?
- Which metric will reveal saturation before an outage spreads?
- Are logs safe by default?
- Can we distinguish validation bugs, dependency failure, and deploy regressions?
