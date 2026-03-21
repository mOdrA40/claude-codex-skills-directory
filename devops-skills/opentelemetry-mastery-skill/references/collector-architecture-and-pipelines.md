# Collector Architecture and Pipelines

## Rules

- Collector topology should be simple enough to understand during failures.
- Agent, gateway, and hybrid designs solve different organizational problems.
- Every processor, exporter, and transform increases operational complexity.
- Backpressure, retry, batching, and memory behavior are part of the system design.

## Architecture Guidance

- Use agents when local enrichment or isolation matters.
- Use gateways when shared processing, policy, or vendor fanout is needed.
- Keep transformation logic controlled and well-owned.
- Design for degraded modes when backends are slow or unavailable.

## Principal Review Lens

- What breaks first if the backend slows down dramatically?
- Which collector pipeline is hardest to debug today?
- Are we using processors to hide upstream instrumentation problems?
- Which topology gives the best balance of autonomy and control?
