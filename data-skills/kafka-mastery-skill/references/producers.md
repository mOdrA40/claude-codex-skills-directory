# Producers (Kafka)

## Rules

- Producer settings define durability, latency, and throughput tradeoffs.
- Batching, compression, acknowledgements, and retries should match business semantics.
- Keys must be chosen with ordering and scale in mind.
- Producer idempotence reduces pain but does not replace end-to-end idempotency.

## Principal Review Lens

- What delivery guarantee does this producer actually provide?
- Is batching helping or hiding latency spikes?
- Which producer setting could amplify duplicates or overload?
