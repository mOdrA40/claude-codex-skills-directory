# Evaluation and Release Guardrails for ML Backends

## Purpose

Offline metrics are necessary but insufficient. A production ML backend needs guardrails that connect model quality, product behavior, operational safety, and rollback readiness.

## Layers of Evaluation

### Offline model evaluation

Examples:

- accuracy
- F1
- ROC-AUC
- perplexity
- retrieval recall
- ranking NDCG

### Slice-based evaluation

Measure by:

- tenant
- language
- geography
- traffic class
- cold-start inputs
- low-frequency edge cases

### Online evaluation

Measure:

- latency percentiles
- failure rate
- business conversion or quality outcomes
- fallback rate
- human override rate
- cost per request

## Bad vs Good: One-Metric Release

```text
❌ BAD
Ship because validation accuracy improved from 0.91 to 0.92.

✅ GOOD
Ship only after checking slice regressions, latency impact, fallback behavior, and rollback readiness.
```

## Regression Gates

Before release, define explicit gates such as:

- no severe regression on protected slices
- p95 latency stays within budget
- memory footprint within serving target
- fallback rate remains below threshold
- prompt or model version is traceable
- canary rollback can be executed within minutes

## LLM-Specific Guardrails

For LLM-backed systems, evaluate:

- hallucination rate by task
- refusal or policy behavior
- prompt injection resistance where applicable
- retrieval grounding quality
- output schema adherence
- token usage and cost ceilings

## Human-in-the-Loop Policy

Some systems should not auto-ship predictions into downstream actions.

Require review when:

- confidence is low
- novelty is high
- protected domain rules are involved
- output affects billing, security, compliance, or irreversible actions

## Release Checklist

- Dataset and model version are fixed.
- Evaluation covers important slices.
- Online SLO impact is estimated.
- Canary plan exists.
- Rollback plan exists.
- Monitoring and alerts are ready.
- Known failure modes are documented.

## Principal Review Questions

- What bad decision can this model make at scale?
- Which user segment is most likely to be harmed by regression?
- What signal tells operators to roll back?
- Which offline win might still be an online loss?
- Is the system safe when confidence is uncertain?
