# Principal ML Platform Decision Matrix

## Purpose

This guide helps choose the right production architecture for ML systems instead of defaulting to the most complex modern stack.

## Decision Axes

- latency sensitivity
- model size and cost
- feature freshness requirements
- explainability requirements
- online vs batch ratio
- safety and policy criticality
- operator maturity

## Choose Simpler Serving When

- classic models or small neural models are enough
- feature pipelines are stable
- latency and cost budgets are tight
- business risk from model complexity is high

## Choose More Complex ML Platforming When

- multiple models and versions must be promoted safely
- online and offline features differ substantially
- RAG, ranking, or LLM workflows need orchestration
- rollback and evaluation governance need central control

## Bad vs Good

```text
❌ BAD
Adopt vector DB + reranker + LLM + feature store + registry + orchestration just because modern ML teams do it.

✅ GOOD
Add platform components only when a specific failure mode, governance need, or scaling constraint justifies them.
```

## Review Questions

- which platform component removes real risk right now?
- what cost and complexity does it add?
- what failure mode becomes easier to detect or control?
