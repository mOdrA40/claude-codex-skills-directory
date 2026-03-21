# Routing, Clusters, and Upstream Policy

## Rules

- Routing policy should reflect service ownership and failure semantics clearly.
- Cluster config, locality, health checking, and endpoint policy all affect user-facing behavior.
- Timeouts and retries belong to contract design, not only to proxy config.
- Keep upstream policy explainable and testable.

## Failure Modes

- Healthy-looking routes sending traffic into unhealthy semantics.
- Locality or load-balancing decisions amplifying partial failure.
- One cluster setting breaking a critical dependency unexpectedly.
- Retry behavior hiding upstream overload until it becomes catastrophic.

## Principal Review Lens

- Which upstream policy is doing the most hidden damage today?
- Can operators predict request behavior during one-zone or one-host impairment?
- Are we modeling service contracts correctly at the proxy?
- What route or cluster default deserves stricter review?
