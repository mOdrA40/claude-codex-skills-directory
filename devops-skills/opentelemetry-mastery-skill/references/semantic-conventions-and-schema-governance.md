# Semantic Conventions and Schema Governance

## Rules

- Telemetry fields are contracts and need governance.
- Standard conventions should be adopted where they improve cross-team comprehension.
- Custom attributes must justify their operational value and cost.
- Schema drift across teams destroys cross-service debugging.

## Governance Model

- Maintain naming guidance for services, operations, environments, versions, and dependency dimensions.
- Review new high-value attributes the way you review public APIs.
- Make deprecations and semantic changes explicit.
- Align traces, metrics, and logs on shared identity dimensions where possible.

## Principal Review Lens

- Which attribute inconsistencies waste the most debugging time today?
- Are teams inventing schema independently because the platform is missing guidance?
- Can a new engineer infer service relationships from telemetry alone?
- What field should be standardized next for maximum leverage?
