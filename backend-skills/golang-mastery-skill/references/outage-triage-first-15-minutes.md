# Outage Triage: First 15 Minutes in Go Services

## First Questions

- did a deploy, flag change, or migration just happen?
- is the bottleneck CPU, DB pool, queue lag, lock contention, or dependency timeout?
- are goroutines, retries, or consumers amplifying pressure?
- is one tenant or one workload shape causing shared pain?

## Containment Options

- halt rollout
- mark unhealthy instances out of rotation
- shed low-priority traffic
- pause toxic consumers
- reduce fan-out and optional enrichments

## Avoid

- restarting blindly
- adding retries without budget and idempotency thinking
- scaling out before knowing the true bottleneck
- changing schema mid-incident unless strictly necessary

## Agent Heuristic

Contain load and protect critical workflows first; optimize and refactor later.
