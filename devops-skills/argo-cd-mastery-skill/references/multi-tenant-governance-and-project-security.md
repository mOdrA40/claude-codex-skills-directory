# Multi-Tenant Governance and Project Security

## Rules

- Shared Argo CD platforms need strict boundaries for projects, repos, clusters, and namespaces.
- App teams should not be able to escape intended scope casually.
- Governance should focus on cluster safety, tenancy, and supportability.
- Repository trust and cluster destination policy must be explicit.

## Practical Guidance

- Standardize projects, destination restrictions, and source repo controls.
- Keep admin versus tenant responsibilities well separated.
- Review high-risk capabilities such as broad namespace or cluster access.
- Make exception paths explicit and time-bounded where possible.

## Principal Review Lens

- Which tenant has too much cluster reach today?
- What project policy is too broad to trust?
- Are we relying on convention more than enforcement?
- What control most reduces shared-platform risk?
