# Production Model Serving

## Purpose

Serving a model in production is a backend systems problem first and an ML problem second. The hardest failures are usually not about tensor math. They are about timeouts, skew, load, retries, contracts, and rollback.

## Serving Contract

Every model-backed endpoint should define:

- input schema
- payload size limits
- versioning rules
- timeout budget
- fallback or degraded behavior
- idempotency expectation
- observability fields
- rollout and rollback behavior

## Architectural Split

Keep these concerns separate:

- request parsing and auth
- feature/preprocessing pipeline
- model invocation
- post-processing / policy layer
- side effects such as logging, persistence, or webhooks

## Bad vs Good: Hidden Preprocessing Drift

```python
# ❌ BAD: training path and serving path normalize text differently.
def train_preprocess(text: str) -> str:
    return text.lower().strip()

def serve_preprocess(text: str) -> str:
    return text.strip()
```

```python
# ✅ GOOD: shared normalization artifact or code path.
def normalize_text(text: str) -> str:
    return text.lower().strip()
```

## Deadlines and Time Budgets

Define an end-to-end SLA and split it intentionally.

Example:

- 50 ms auth and parsing
- 120 ms feature generation
- 250 ms model inference
- 50 ms policy/post-processing
- 30 ms response encoding

If no budget exists, every dependency will consume the whole request.

## Failure Modes

Classify failures into:

- invalid input
- dependency timeout
- model artifact missing
- feature lookup unavailable
- model execution failed
- policy rejection
- system overload

Map these to stable external semantics.

## Backpressure and Overload

When the service is overloaded:

- reject early instead of queueing forever
- shed optional enrichment first
- protect core endpoints from batch traffic
- isolate slow or expensive model classes
- define concurrency caps for GPU or CPU-bound workloads

## Rollout Strategy

Use one of these intentionally:

- shadow traffic
- canary by percentage
- tenant-based canary
- offline replay before release
- feature-flagged model switching

Never replace a production model with a new one only because offline accuracy looked better.

## Observability Fields

At minimum capture:

- request ID
- model version
- feature version
- prompt version if applicable
- latency by phase
- fallback used or not
- response class / status
- cost or token usage if relevant

## Review Checklist

- Serving and training preprocessing are identical.
- Time budgets are explicit.
- Concurrency limits exist.
- Rollback is defined.
- Model version is observable.
- Failure categories are stable and actionable.
