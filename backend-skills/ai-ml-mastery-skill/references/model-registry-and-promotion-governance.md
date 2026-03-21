# Model Registry and Promotion Governance

## Principle

A model registry is not just a storage layer for artifacts. It is the control point for promotion rules, rollback safety, lineage, and operational trust.

## Rules

- promotion gates must be explicit
- registry entries should bind model, config, tokenizer, feature version, and evaluation metadata
- stage names mean nothing unless promotion policy is enforced consistently
- rollback must be fast and observable

## Review Questions

- what evidence is required before promotion?
- can operators identify exactly which artifact version is live?
- can a rollback happen without retraining or rebuilding?
