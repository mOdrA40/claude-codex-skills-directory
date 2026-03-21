# Evaluation Harness Design for ML Systems

## Principle

A model is only as trustworthy as the harness used to evaluate it. Ad hoc evaluation creates false confidence and unsafe promotion decisions.

## Rules

- define standard datasets and slices
- keep evaluation code versioned and reproducible
- separate benchmark suites by use case
- include operational metrics such as latency and cost, not only quality metrics
- make evaluation artifacts easy to compare across versions

## Review Questions

- would two engineers evaluate the same candidate the same way?
- what slice hides the biggest risk?
- does the harness measure both model quality and serving reality?
