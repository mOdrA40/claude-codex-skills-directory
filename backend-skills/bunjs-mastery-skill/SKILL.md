---
 name: bunjs-docker-mastery
 description: |
  Principal/Senior-level Bun.js playbook for backend APIs, runtime-aware TypeScript, Docker delivery, reliability, observability, testing, and production operations.
  Use when: building or reviewing Bun services, hardening Hono-based APIs, modernizing Node-style backends for Bun, improving Docker and CI quality, or preparing Bun apps for production.
---

# Bun.js + Docker Mastery

## Operate

- Confirm the goal, scope, Bun version, deployment target, Docker constraints, database choice, traffic profile, and definition of done.
- Prefer small vertical slices with tests and explicit tradeoffs.
- Use Bun-native capabilities when they simplify the system, but do not force Bun-specific APIs where the standard Web platform is already clear.
- Optimize for operability: graceful shutdown, structured logs, health checks, timeouts, and safe defaults are part of the baseline.

> The goal is not just “fast on benchmarks”. The goal is a service that stays easy to debug and safe to run in production.

## Default Standards

- Keep `src/index.ts` as bootstrap only; move app wiring to `src/app.ts` and business logic to services/use-cases.
- Validate environment variables and request payloads at the boundary.
- Prefer explicit error types and one global error mapping strategy.
- Keep TypeScript strict; avoid `any`, hidden type assertions, and implicit runtime contracts.
- Prefer idempotent handlers for side-effecting endpoints where retries can happen.
- Treat database, cache, and outbound HTTP as failure-prone dependencies: always define timeouts and degradation behavior.

## “Bad vs Good” (common production pitfalls)

```typescript
// ❌ BAD: parsing untrusted input directly in the service layer.
const user = await userService.create(await c.req.json())

// ✅ GOOD: validate at the HTTP boundary before entering business logic.
const payload = c.req.valid("json")
const user = await userService.create(payload)
```

```typescript
// ❌ BAD: fire-and-forget async work with no ownership or logging.
void sendWebhook(order)

// ✅ GOOD: await or queue the side effect with explicit failure handling.
await webhookPublisher.publish(order).catch((error) => {
  logger.error({ error, orderId: order.id }, "publish webhook failed")
  throw new AppError("Webhook publish failed", 503, "WEBHOOK_UNAVAILABLE")
})
```

## Recommended Structure

```text
src/
├── index.ts
├── app.ts
├── config/
├── routes/
├── controllers/
├── services/
├── repositories/
├── middlewares/
├── utils/
└── types/
```

- Keep `index.ts` for bootstrap only; initialize config, logger, app, and shutdown wiring there.
- Keep framework-specific request/response types near the transport boundary.
- Keep service and repository layers free from Hono/Elysia context objects.
- Prefer one explicit composition root instead of hidden module-level singletons.

## Workflow (Feature / Refactor / Bug)

1. Reproduce the behavior or encode it in tests.
2. Decide boundaries: transport, validation, use-case, persistence, outbound integrations.
3. Define runtime behavior: timeout, retry, cancellation, overload, and shutdown posture.
4. Implement the smallest end-to-end slice.
5. Validate tests, type checks, linting, and container build if deployment uses Docker.
6. Review observability, rollout safety, and dependency-failure behavior before release.

## Validation Commands

- Run `bun install --frozen-lockfile`.
- Run `bun run test`.
- Run `bunx tsc --noEmit` if the project uses standalone TypeScript checks.
- Run `bunx @biomejs/biome check .`.
- Run `bun run build` before release.
- Run Docker build validation for production images when Docker is part of the deliverable.

## Runtime and API Guardrails

- Use a single request ID / trace ID strategy and include it in logs and error responses where appropriate.
- Set request body limits, timeouts, and rate limits for public endpoints.
- Do not leak stack traces, secrets, or database errors to clients.
- Prefer `Bun.password` for password hashing instead of legacy `bcrypt` stacks unless compatibility requires otherwise.
- Make shutdown explicit: stop accepting traffic, drain in-flight work, and close DB/Redis clients.
- Treat all outbound IO as failure-prone: set deadlines, connection limits, and fallback behavior.
- Separate online request traffic from background jobs and expensive fan-out work.
- Be explicit about file upload limits, streaming boundaries, and webhook authenticity checks.

## Docker & Deployment Defaults

- Use multi-stage Docker builds.
- Run as non-root.
- Pin Bun base image versions; avoid floating `latest` tags.
- Add `/health` and `/ready` endpoints for orchestration environments.
- Keep image contents minimal and deterministic.

## Testing Defaults

- Unit tests for pure logic and service policies.
- Integration tests for database repositories and route behavior.
- E2E tests only for critical user flows.
- Keep mocks small and close to the consumer boundary.

## Code Review Checklist

- Confirm request validation, config validation, and response mapping happen at boundaries.
- Confirm services do not depend directly on web framework context or global mutable state.
- Confirm all external dependencies have timeout and failure semantics defined.
- Confirm structured logs, health/readiness endpoints, and shutdown hooks exist for deployable services.
- Confirm Docker images and CI steps are reproducible, pinned, and non-root where applicable.

## Decision Heuristics

- Choose Bun when JavaScript/TypeScript productivity matters and the workload is primarily IO-bound.
- Prefer Bun when startup time, single-runtime simplicity, and modern Web APIs improve delivery speed without sacrificing operability.
- Prefer another stack when the service is CPU-heavy, needs a very mature enterprise framework ecosystem, or depends on packages that are still Node-only in practice.

## References

- Clean code patterns: [references/clean-code-patterns.md](references/clean-code-patterns.md)
- Agent instructions for backend tasks: [references/agent-instructions-for-backend-tasks.md](references/agent-instructions-for-backend-tasks.md)
- Backend cost and performance tradeoffs: [references/backend-cost-performance-tradeoffs.md](references/backend-cost-performance-tradeoffs.md)
- Dependency failure decision tree: [references/dependency-failure-decision-tree.md](references/dependency-failure-decision-tree.md)
- Debugging guide: [references/debugging-guide.md](references/debugging-guide.md)
- Library arsenal: [references/library-arsenal.md](references/library-arsenal.md)
- Bun architecture decision framework: [references/bun-architecture-decision-framework.md](references/bun-architecture-decision-framework.md)
- Production decision matrix: [references/production-decision-matrix.md](references/production-decision-matrix.md)
- Framework selection: [references/framework-selection.md](references/framework-selection.md)
- Edge vs server runtime tradeoffs: [references/edge-vs-server-runtime-tradeoffs.md](references/edge-vs-server-runtime-tradeoffs.md)
- Hono API patterns: [references/hono-api-patterns.md](references/hono-api-patterns.md)
- Hono middleware governance: [references/hono-middleware-governance.md](references/hono-middleware-governance.md)
- ElysiaJS patterns: [references/elysiajs-patterns.md](references/elysiajs-patterns.md)
- Elysia plugin risk management: [references/elysia-plugin-risk-management.md](references/elysia-plugin-risk-management.md)
- Auth and session security: [references/auth-and-session-security.md](references/auth-and-session-security.md)
- Background jobs and queues: [references/background-jobs-and-queues.md](references/background-jobs-and-queues.md)
- Background processing and consumers: [references/background-processing-and-consumers.md](references/background-processing-and-consumers.md)
- Contract validation and response mapping: [references/contract-validation-and-response-mapping.md](references/contract-validation-and-response-mapping.md)
- Capacity planning and load shedding: [references/capacity-planning-and-load-shedding.md](references/capacity-planning-and-load-shedding.md)
- Dependency timeouts and abort strategy: [references/dependency-timeouts-and-abort-strategy.md](references/dependency-timeouts-and-abort-strategy.md)
- Schema contracts and validation: [references/schema-contracts-and-validation.md](references/schema-contracts-and-validation.md)
- Database and transactions: [references/database-and-transactions.md](references/database-and-transactions.md)
- API evolution and versioning: [references/api-evolution-and-versioning.md](references/api-evolution-and-versioning.md)
- Drizzle and migration safety: [references/drizzle-and-migrations.md](references/drizzle-and-migrations.md)
- Observability: [references/observability.md](references/observability.md)
- Bun runtime operability: [references/bun-runtime-operability.md](references/bun-runtime-operability.md)
- Realtime and WebSocket patterns: [references/realtime-and-websocket-patterns.md](references/realtime-and-websocket-patterns.md)
- Security and outbound HTTP: [references/security-and-outbound-http.md](references/security-and-outbound-http.md)
- SLO, error budget, and release gates: [references/slo-error-budget-and-release-gates.md](references/slo-error-budget-and-release-gates.md)
- Testing strategy: [references/testing-strategy.md](references/testing-strategy.md)
- Webhooks, idempotency, and retries: [references/webhooks-idempotency-and-retries.md](references/webhooks-idempotency-and-retries.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident response playbook: [references/incident-response-playbook.md](references/incident-response-playbook.md)
- Multi-tenant rate limits: [references/multi-tenant-and-rate-limits.md](references/multi-tenant-and-rate-limits.md)
- Operational smells and red flags: [references/operational-smells-and-red-flags.md](references/operational-smells-and-red-flags.md)
- Outage triage, first 15 minutes: [references/outage-triage-first-15-minutes.md](references/outage-triage-first-15-minutes.md)
- Principal backend code review playbook: [references/principal-backend-code-review-playbook.md](references/principal-backend-code-review-playbook.md)
- Queue poison message and dead-letter playbook: [references/queue-poison-message-and-dead-letter-playbook.md](references/queue-poison-message-and-dead-letter-playbook.md)
- Review checklists by change type: [references/review-checklists-by-change-type.md](references/review-checklists-by-change-type.md)
- Safe rollout patterns for high-traffic services: [references/safe-rollout-patterns-for-high-traffic-services.md](references/safe-rollout-patterns-for-high-traffic-services.md)
- Service decomposition and boundary decisions: [references/service-decomposition-and-boundary-decisions.md](references/service-decomposition-and-boundary-decisions.md)
- Tenant fairness and noisy neighbor playbook: [references/tenant-fairness-and-noisy-neighbor-playbook.md](references/tenant-fairness-and-noisy-neighbor-playbook.md)
- Zero-downtime schema and contract migrations: [references/zero-downtime-schema-and-contract-migrations.md](references/zero-downtime-schema-and-contract-migrations.md)

## Scripts & Assets

- `scripts/init-project.sh` - initialize a Bun project from the template.
- `scripts/healthcheck.ts` - health endpoint template.
- `assets/project-template/` - project boilerplate.
