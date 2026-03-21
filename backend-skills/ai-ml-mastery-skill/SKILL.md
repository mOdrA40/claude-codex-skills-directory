---
name: ai-ml-principal-engineer
description: |
  Principal/Senior-level AI/ML playbook for production machine learning systems, LLM-enabled backends, model serving, training pipelines, evaluation discipline, reliability, security, and MLOps.
  Use when: designing ML services, building or reviewing training/inference code, selecting model architectures, fine-tuning transformers, hardening model APIs, debugging performance or correctness issues, or preparing ML systems for production.
---

# AI/ML Mastery (Senior → Principal)

## Operate

- Start by confirming: objective, success metric, data availability, privacy/security constraints, latency and throughput targets, compute budget, deployment target, and the definition of done.
- Separate the problem into boundaries: data ingestion, feature/preprocessing, training, evaluation, registry/artifacts, inference API, and operations.
- Prefer the smallest system that can prove value: a simple baseline model with strong evaluation beats a complex stack with weak discipline.
- Treat ML work as software engineering: reproducibility, observability, rollback, and failure handling are part of the feature.

> The goal is not just a high offline metric. The goal is a model-backed backend that is correct, measurable, operable, and safe in production.

## Default Standards

- Keep notebooks for exploration only; production logic belongs in versioned Python modules and tests.
- Validate schema, dtypes, ranges, nullability, and label quality at the data boundary.
- Make training and inference preprocessing identical by sharing explicit pipeline code.
- Prefer typed config objects and immutable runtime settings.
- Use structured logging and explicit error taxonomy for data, model, dependency, and serving failures.
- Define latency budgets, timeout behavior, fallback behavior, and model version strategy before exposing public inference endpoints.
- Default to simpler baselines before large models; earn complexity with measured gains.

## “Bad vs Good” (common production pitfalls)

```python
# ❌ BAD: training and inference use different preprocessing.
train_text = text.lower().strip()
serve_text = text.strip()

# ✅ GOOD: one shared preprocessing pipeline used everywhere.
normalized_text = text_normalizer.normalize(text)
```

```python
# ❌ BAD: silent fallback hides model loading failures.
try:
    model = load_model(path)
except Exception:
    model = None

# ✅ GOOD: fail explicitly or switch to a known degraded mode.
try:
    model = load_model(path)
except FileNotFoundError as error:
    raise ModelBootstrapError(f"model artifact missing: {path}") from error
```

```python
# ❌ BAD: unbounded inference call with no deadline.
prediction = client.predict(payload)

# ✅ GOOD: explicit deadline and graceful failure mapping.
prediction = client.predict(payload, timeout=2.0)
```

## Workflow (Feature / Refactor / Bug)

1. Define the business outcome, online/offline metrics, and failure tolerance.
2. Establish a reproducible baseline and dataset contract.
3. Design boundaries between training code, model packaging, and serving code.
4. Implement the smallest end-to-end slice with tests and evaluation reports.
5. Validate reproducibility, security, performance, and rollback readiness.
6. Ship with monitoring for latency, throughput, drift, quality, and cost.

## Validation Commands

- Run `python -m pytest`.
- Run `python -m ruff check .` if Ruff is used.
- Run `python -m mypy src` for typed code paths when the repo uses MyPy.
- Run `python -m pytest -k inference` for serving-critical tests.
- Run `python -m pytest --maxfail=1 --disable-warnings` during local debugging.
- Run smoke evaluation for the current model artifact before release.
- Run container build validation if inference is deployed via Docker.

## Backend-Oriented ML Guardrails

- Always version models, prompts, tokenizer assets, and preprocessing artifacts together.
- Do not call external model providers from request paths without timeouts, retries, budgets, and fallback behavior.
- Separate online inference from heavy offline batch jobs.
- Prefer async queue-based processing for expensive enrichment, reranking, or embedding backfills.
- Protect inference endpoints with payload size limits, authn/authz, and rate limiting.
- Log request IDs, model version, feature version, and decision metadata without leaking raw sensitive payloads.

## Decision Framework: Library Selection

| Task | Default Choice | Use Alternative When |
|------|----------------|----------------------|
| Deep learning training | PyTorch | TensorFlow for TPU-heavy production, JAX for research-heavy experimentation |
| Classical/tabular ML | scikit-learn | XGBoost/LightGBM for stronger tabular baselines, CatBoost for categorical-heavy data |
| LLM application layer | transformers + sentence-transformers | vLLM for high-throughput serving, llama.cpp for edge or constrained environments |
| Data processing | pandas | polars for larger columnar workloads, dask/spark for distributed pipelines |
| Experiment tracking | MLflow | Weights & Biases or Neptune when team workflows require hosted collaboration |
| Hyperparameter tuning | Optuna | Ray Tune when you need distributed search orchestration |

## Architecture Selection Heuristics

```text
Text classification          -> DistilBERT for speed, RoBERTa for stronger accuracy
Embeddings / retrieval       -> sentence-transformers or hosted embedding APIs with evaluation gates
Vision classification        -> ResNet/EfficientNet as baseline, ViT when data and budget justify it
Object detection             -> YOLO for speed, DETR/RT-DETR when workflow favors transformer-based designs
Tabular prediction           -> Logistic regression / XGBoost baseline first, deep tabular only if proven necessary
Recommendation               -> retrieval + ranking pipelines, not a single monolithic model by default
Time series                  -> statistical baseline first, then TFT/PatchTST when complexity is justified
```

## Recommended Project Structure

```text
project/
├── pyproject.toml
├── README.md
├── src/
│   └── app/
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── serving/
│       └── observability/
├── tests/
├── scripts/
├── configs/
├── notebooks/
└── docker/
```

## Reliability, Security, and Operations

- Make model bootstrap behavior explicit: fail closed, fail open, or degraded mode.
- Bound input sizes, token counts, image dimensions, and recursion depth for untrusted requests.
- Prefer queue-based retries over client-side blind retries for expensive inference.
- Track feature drift, data freshness, and serving skew between training and production.
- Keep PII out of prompts, logs, traces, and experiment artifacts unless explicitly required and governed.
- Store secrets and provider credentials in secret managers, never in notebooks or source files.

## Training and Evaluation Checklist

- [ ] Define offline and online success metrics before training
- [ ] Fix random seeds when reproducibility matters
- [ ] Check train/validation/test leakage
- [ ] Validate preprocessing parity between train and serve
- [ ] Save model artifact, config, tokenizer, and feature metadata together
- [ ] Record dataset version and experiment version
- [ ] Benchmark latency, throughput, memory, and cost
- [ ] Define rollback or model disable strategy before release

## References

- Deep learning systems: [references/deep-learning.md](references/deep-learning.md)
- Transformers and LLMs: [references/transformers-llm.md](references/transformers-llm.md)
- Computer vision: [references/computer-vision.md](references/computer-vision.md)
- Classical machine learning: [references/machine-learning.md](references/machine-learning.md)
- NLP systems: [references/nlp.md](references/nlp.md)
- MLOps and deployment: [references/mlops.md](references/mlops.md)
- Feature stores and data contracts: [references/feature-stores-and-data-contracts.md](references/feature-stores-and-data-contracts.md)
- Model registry and promotion governance: [references/model-registry-and-promotion-governance.md](references/model-registry-and-promotion-governance.md)
- Principal ML platform decision matrix: [references/principal-ml-platform-decision-matrix.md](references/principal-ml-platform-decision-matrix.md)
- Production model serving: [references/production-serving.md](references/production-serving.md)
- Batch vs online inference: [references/batch-vs-online-inference.md](references/batch-vs-online-inference.md)
- Drift detection and observability: [references/drift-detection-and-observability.md](references/drift-detection-and-observability.md)
- Evaluation and release guardrails: [references/evaluation-and-guardrails.md](references/evaluation-and-guardrails.md)
- Evaluation harness design: [references/evaluation-harness-design.md](references/evaluation-harness-design.md)
- Incident response for ML systems: [references/incident-response-for-ml-systems.md](references/incident-response-for-ml-systems.md)
- Retrieval and RAG systems: [references/retrieval-and-rag-systems.md](references/retrieval-and-rag-systems.md)
- Inference reliability and cost control: [references/inference-reliability-and-cost.md](references/inference-reliability-and-cost.md)
- LLM safety and adversarial inputs: [references/llm-safety-and-adversarial-inputs.md](references/llm-safety-and-adversarial-inputs.md)
