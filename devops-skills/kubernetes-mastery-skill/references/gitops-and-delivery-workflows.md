# GitOps and Delivery Workflows

## Rules

- Delivery workflow should preserve auditability and rollback clarity.
- GitOps is not valuable if it hides ownership and intent.
- Promotion between environments should be explicit and reviewable.
- Drift detection and emergency change policy must coexist.

## Principal Review Lens

- Who owns production truth when Git and cluster differ?
- Can emergency changes be reconciled safely?
- Is rollout visibility good enough for fast rollback?
