# Producers (Kafka)

## Rules

- Producer settings define durability, latency, and throughput tradeoffs.
- Batching, compression, acknowledgements, and retries should match business semantics.
- Keys must be chosen with ordering and scale in mind.
- Producer idempotence reduces pain but does not replace end-to-end idempotency.

## Producer Heuristics

### Producer config is part of the delivery contract

Acknowledgements, batching, linger, retries, and idempotence together shape what the system is really promising about latency, durability, and duplicate risk.

### Producers should be designed for pressure, not just happy-path throughput

The key question is how producer behavior changes when brokers slow down, partitions become hot, or downstream expectations become stricter.

### Key choice and producer behavior are linked

A producer that chooses unstable or poorly distributed keys can sabotage scale and ordering predictability even if its low-level settings are correct.

## Common Failure Modes

### Config by folklore

Teams copy producer settings from another service or tutorial without validating whether the durability and latency tradeoff still fits their business path.

### Batching hides user pain

Throughput improves on paper, but burstier latency and slower failure visibility make user-facing behavior worse.

### Idempotence overtrusted

Producer idempotence is enabled, but teams act as if the whole workflow is now duplicate-safe when downstream effects still are not.

## Principal Review Lens

- What delivery guarantee does this producer actually provide?
- Is batching helping or hiding latency spikes?
- Which producer setting could amplify duplicates or overload?
- Which producer assumption becomes least safe under broker slowdown or retry pressure?
