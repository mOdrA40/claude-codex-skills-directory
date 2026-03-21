# Ingestion, Batching, and Scale Behavior

## Rules

- Trace ingest is bursty and must be designed with headroom.
- Batching and buffering decisions affect latency, durability, and cost together.
- Sampling-aware expectations should shape ingestion capacity planning.
- Collector and backend behavior must be understood end-to-end.

## Practical Guidance

- Model peak incident ingest, not only steady state.
- Watch queueing, drops, flush timing, and backend write behavior.
- Distinguish collector-side loss from Tempo-side bottlenecks.
- Plan for partial backpressure and degraded-mode operation.

## Principal Review Lens

- What fails first under 2x or 5x incident trace volume?
- Are we losing traces before the backend or inside it?
- Which buffering choice most risks hidden truth loss?
- Can operators explain ingest health quickly under pressure?
