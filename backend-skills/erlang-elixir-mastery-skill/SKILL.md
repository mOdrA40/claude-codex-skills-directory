---
name: erlang-elixir-principal-engineer
description: |
  Principal/Senior-level Erlang and Elixir playbook for highly available backend systems, OTP design, actor-model concurrency, reliability, observability, and production operations.
  Use when: building or reviewing Elixir or Erlang services, designing OTP supervision trees, handling high-concurrency workloads, debugging distributed failures, improving resilience, or preparing BEAM systems for production.
---

# Erlang / Elixir Mastery (Senior → Principal)

## Operate

- Start by confirming: language choice (Erlang or Elixir), OTP version, cluster topology, persistence strategy, workload shape, availability goals, and the definition of done.
- Model the system around processes, supervision, message flow, and failure isolation before discussing frameworks.
- Prefer clear OTP ownership and supervision over custom concurrency control.
- Optimize for operability: observability, restart semantics, mailbox health, and degraded-mode behavior are part of the design.

> The goal is not to show off the actor model. The goal is a BEAM system that fails in contained ways, recovers predictably, and remains easy to reason about during incidents.

## Default Standards

- Use OTP behaviors (`GenServer`, `Supervisor`, `DynamicSupervisor`, `GenStage` when justified) instead of ad-hoc process orchestration.
- Keep Phoenix or HTTP layers thin; move policy and orchestration into testable modules.
- Define message contracts and state transitions explicitly.
- Treat mailbox growth, process cardinality, and backpressure as primary production risks.
- Prefer supervision trees and retries at the right layer over defensive rescue blocks everywhere.
- Make timeout, retry, and circuit breaker behavior explicit for outbound dependencies.
- Use telemetry and structured logging consistently.

## “Bad vs Good” (common production pitfalls)

```elixir
# ❌ BAD: using a GenServer as a catch-all service locator.
def handle_call({:do_everything, payload}, _from, state) do
  result = BigBallOfMud.run(payload, state)
  {:reply, result, state}
end

# ✅ GOOD: keep the process focused and delegate domain behavior.
def handle_call({:create_user, attrs}, _from, state) do
  case Users.create(attrs, state.deps) do
    {:ok, user} -> {:reply, {:ok, user}, state}
    {:error, reason} -> {:reply, {:error, reason}, state}
  end
end
```

```elixir
# ❌ BAD: unbounded Task spawning in a request path.
Enum.each(events, fn event ->
  Task.start(fn -> publish(event) end)
end)

# ✅ GOOD: use supervised, bounded concurrency.
Task.Supervisor.async_stream_nolink(MyApp.TaskSupervisor, events, &publish/1, max_concurrency: 8, timeout: 5_000)
|> Stream.run()
```

```elixir
# ❌ BAD: catch-all rescue hides operational signals.
try do
  external_call()
rescue
  _ -> :ok
end

# ✅ GOOD: classify the failure and surface it.
case external_call() do
  {:ok, result} -> {:ok, result}
  {:error, :timeout} -> {:error, :dependency_timeout}
  {:error, reason} -> {:error, {:dependency_failed, reason}}
end
```

## Workflow (Feature / Refactor / Bug)

1. Reproduce the issue or encode the expected behavior in tests.
2. Decide process ownership, supervision strategy, and message boundaries.
3. Define failure semantics: restart, retry, drop, dead-letter, or escalate.
4. Implement the smallest end-to-end slice.
5. Validate mailbox behavior, process counts, timeout paths, and telemetry.
6. Review cluster behavior, deploy safety, and rollback readiness.

## Validation Commands

- Run `mix format --check-formatted`.
- Run `mix test`.
- Run `mix credo --strict` if Credo is used.
- Run `mix dialyzer` for type/spec analysis when configured.
- Run `mix test --trace` during debugging.
- Run `mix phx.routes` and endpoint smoke checks for Phoenix applications.
- Run release smoke tests if the application ships as an OTP release.

## Recommended Service Shape

```text
lib/my_app/
├── application.ex
├── telemetry.ex
├── domain/
│   ├── users.ex
│   └── errors.ex
├── services/
│   └── create_user.ex
├── adapters/
│   ├── repo.ex
│   ├── http_client.ex
│   └── queue_publisher.ex
└── web/
    ├── router.ex
    ├── controllers/
    └── plugs/
```

- Keep `application.ex` focused on supervision and startup wiring.
- Keep Phoenix controllers, channels, or plugs thin; move orchestration into domain/service modules.
- Make process ownership explicit: know which module owns a queue, cache, subscription, or outbound client.
- Prefer named responsibilities over generic “server” modules that accumulate unrelated state.

## OTP and Concurrency Guardrails

- Every long-lived process should have a clear owner and supervision policy.
- Avoid turning one GenServer into a bottleneck for unrelated responsibilities.
- Bound concurrency for fan-out work; use supervised tasks or queues.
- Watch mailbox growth, reduction spikes, and scheduler imbalance.
- Prefer message passing over shared mutable escape hatches such as ETS misuse.
- Use ETS intentionally: great for read-heavy shared data, dangerous as hidden global state.

## Service and API Defaults

- Validate input at the boundary before it reaches domain logic.
- Map domain errors consistently to HTTP, gRPC, or queue semantics.
- Use idempotency keys for retried commands with side effects.
- Set explicit client timeouts and pool limits for outbound HTTP/database calls.
- Do not leak stack traces or internal exception details to clients.

## Reliability, Distributed Systems, and Operations

- Design supervision trees around blast radius, not just code organization.
- Decide what must restart together and what must fail independently.
- Be explicit about node discovery, cookie management, and cluster partitions.
- Treat netsplits, mailbox buildup, and dependency slowness as first-class failure modes.
- Expose health checks, readiness, telemetry, and trace correlation for incident response.
- Make overload posture explicit: backpressure, queueing, dropping, or shedding must be intentional.

## Debugging and Incident Heuristics

- Inspect mailbox size, scheduler utilization, and restart intensity before assuming the bug is in business logic.
- Differentiate between one noisy process, one bad supervision subtree, and whole-node resource pressure.
- When a BEAM system looks “slow”, ask whether the bottleneck is message backlog, dependency latency, serialization, or DB pool contention.
- Use telemetry, logs, and trace correlation to connect inbound requests, spawned work, and downstream failures.

## Security Checklist (Minimum)

- Keep secrets out of logs, process state dumps, and crash reports.
- Validate payload size and shape for all external inputs.
- Use least-privilege credentials and role separation.
- Harden admin endpoints, LiveDashboard, and remote shell access.
- Audit distributed node trust boundaries before enabling clustering across networks.

## Code Review Checklist

- Confirm each long-lived process has a clear responsibility, owner, and supervision policy.
- Confirm mailbox growth and fan-out concurrency are bounded for the expected traffic shape.
- Confirm controllers, channels, and consumers delegate domain work instead of becoming orchestration blobs.
- Confirm dependency failures map to explicit timeout/retry/degradation semantics.
- Confirm telemetry and incident signals are sufficient to debug a production failure without attaching a shell first.

## Decision Heuristics

```text
Choose Erlang/Elixir when:
- the workload is highly concurrent and failure isolation matters
- soft real-time behavior and uptime are core requirements
- supervision and message passing are natural fits for the domain
- the team benefits from OTP and BEAM operational strengths

Prefer another backend stack when:
- the problem is simple CRUD with no resilience or concurrency pressure
- ecosystem maturity for a required niche library is missing
- the team cannot support BEAM operational knowledge yet
```

## References

- OTP design and supervision boundaries: [references/otp-design.md](references/otp-design.md)
- Service boundaries and domain design: [references/service-boundaries-and-domain-design.md](references/service-boundaries-and-domain-design.md)
- Agent instructions for backend tasks: [references/agent-instructions-for-backend-tasks.md](references/agent-instructions-for-backend-tasks.md)
- API and event schema compatibility matrix: [references/api-event-schema-compatibility-matrix.md](references/api-event-schema-compatibility-matrix.md)
- Backend cost and performance tradeoffs: [references/backend-cost-performance-tradeoffs.md](references/backend-cost-performance-tradeoffs.md)
- Dependency failure decision tree: [references/dependency-failure-decision-tree.md](references/dependency-failure-decision-tree.md)
- Failure stories and recovery patterns: [references/failure-stories-and-recovery-patterns.md](references/failure-stories-and-recovery-patterns.md)
- Ecto transactions and outbox: [references/ecto-transactions-and-outbox.md](references/ecto-transactions-and-outbox.md)
- Phoenix and service boundaries: [references/phoenix-and-boundaries.md](references/phoenix-and-boundaries.md)
- Broadway, GenStage, and consumer pipelines: [references/broadway-genstage-and-consumers.md](references/broadway-genstage-and-consumers.md)
- Distributed systems and operations: [references/distributed-systems-and-operations.md](references/distributed-systems-and-operations.md)
- Mailbox pressure and backpressure: [references/mailbox-pressure-and-backpressure.md](references/mailbox-pressure-and-backpressure.md)
- Multi-region and failover strategy: [references/multi-region-and-failover-strategy.md](references/multi-region-and-failover-strategy.md)
- Multi-tenant BEAM systems: [references/multi-tenant-beam-systems.md](references/multi-tenant-beam-systems.md)
- Graceful degradation and dependency failure: [references/graceful-degradation-and-dependency-failure.md](references/graceful-degradation-and-dependency-failure.md)
- Telemetry and observability: [references/telemetry-and-observability.md](references/telemetry-and-observability.md)
- Operational smells and red flags: [references/operational-smells-and-red-flags.md](references/operational-smells-and-red-flags.md)
- Outage triage, first 15 minutes: [references/outage-triage-first-15-minutes.md](references/outage-triage-first-15-minutes.md)
- Backend incident postmortem patterns: [references/backend-incident-postmortem-patterns.md](references/backend-incident-postmortem-patterns.md)
- High-risk change preflight checklist: [references/high-risk-change-preflight-checklist.md](references/high-risk-change-preflight-checklist.md)
- Principal backend code review playbook: [references/principal-backend-code-review-playbook.md](references/principal-backend-code-review-playbook.md)
- Queue poison message and dead-letter playbook: [references/queue-poison-message-and-dead-letter-playbook.md](references/queue-poison-message-and-dead-letter-playbook.md)
- Request path vs background path separation: [references/request-path-vs-background-path-separation.md](references/request-path-vs-background-path-separation.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
- Review checklists by change type: [references/review-checklists-by-change-type.md](references/review-checklists-by-change-type.md)
- Safe rollout patterns for high-traffic services: [references/safe-rollout-patterns-for-high-traffic-services.md](references/safe-rollout-patterns-for-high-traffic-services.md)
- Service decomposition and boundary decisions: [references/service-decomposition-and-boundary-decisions.md](references/service-decomposition-and-boundary-decisions.md)
- Stateful vs stateless service decisions: [references/stateful-vs-stateless-service-decisions.md](references/stateful-vs-stateless-service-decisions.md)
- Tenant fairness and noisy neighbor playbook: [references/tenant-fairness-and-noisy-neighbor-playbook.md](references/tenant-fairness-and-noisy-neighbor-playbook.md)
- Zero-downtime schema and contract migrations: [references/zero-downtime-schema-and-contract-migrations.md](references/zero-downtime-schema-and-contract-migrations.md)
