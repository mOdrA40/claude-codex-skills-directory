# Feature Stores and Data Contracts

## Purpose

Many ML production failures come from data contract drift, not model architecture. If online and offline features diverge, the system becomes unpredictable no matter how strong the model is.

## Rules

- define feature schema and ownership explicitly
- version transformations and feature definitions
- distinguish online-serving features from offline-training features
- validate freshness, nullability, and allowed ranges
- track skew between offline and online pipelines

## Bad vs Good

```text
❌ BAD
Training uses one feature derivation path while serving reconstructs features differently in application code.

✅ GOOD
Feature definitions are versioned and shared or validated across training and serving boundaries.
```

## Review Questions

- who owns feature definitions?
- how is skew detected?
- what happens if feature freshness degrades?
- can an old model read new features safely?
