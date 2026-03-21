# Inference Reliability and Cost Control

## Purpose

ML systems often fail not because they are inaccurate, but because they are too slow, too expensive, too fragile, or too opaque under real traffic.

## Reliability Dimensions

For any inference service, monitor and govern:

- p50/p95/p99 latency
- timeout rate
- error rate by dependency
- fallback rate
- queue delay
- memory and accelerator saturation
- cost per request
- token usage if applicable

## Cost Is a Production Metric

For LLM or heavy inference systems, cost must be treated as a first-class control variable.

Examples of cost controls:

- route easy traffic to smaller models
- use cached embeddings or cached responses where safe
- cap output tokens
- reduce reranking depth on degraded mode
- move non-urgent inference off the synchronous path

## Bad vs Good: Hidden Cost Explosion

```text
❌ BAD
A prompt or retrieval change doubles token usage and nobody notices until the bill arrives.

✅ GOOD
Token and cost metrics are tracked per endpoint, version, and tenant class.
```

## Reliability Guardrails

- define concurrency ceilings
- protect expensive models from unbounded fan-in
- use per-tenant or per-feature budgets when relevant
- separate batch and online traffic
- make degraded mode explicit

## Degraded Modes

Examples:

- skip optional enrichment
- fall back to smaller model
- return partial ranking without reranker
- return cached or stale-safe result
- reject low-priority traffic early

## Review Checklist

- Cost is measured by endpoint and version.
- Concurrency is bounded.
- Degraded mode exists.
- Batch traffic cannot starve online traffic.
- Operators know when to shed cost or latency first.
