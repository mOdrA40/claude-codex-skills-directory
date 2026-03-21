# Batch vs Online Inference Architecture

## Principle

Not all inference belongs on a synchronous request path. Mixing online and batch concerns creates latency, cost, and reliability failures.

## Online Inference

Use when:

- low-latency decisions are required
- user-facing actions depend immediately on the result
- fallback behavior is defined

## Batch Inference

Use when:

- throughput matters more than single-request latency
- backfills, enrichment, reranking prep, or nightly scoring are acceptable
- retry and replay economics favor asynchronous execution

## Bad vs Good

```text
❌ BAD
A user-facing API performs heavy enrichment, retrieval, reranking, and generation synchronously for all traffic.

✅ GOOD
The system separates urgent online decisions from background or precomputed inference work.
```

## Review Questions

- which paths truly need synchronous inference?
- what should be precomputed?
- can degraded mode skip expensive steps safely?
