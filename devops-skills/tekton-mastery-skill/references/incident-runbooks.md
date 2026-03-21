# Incident Runbooks (Tekton)

## Cover at Minimum

- Controller degradation.
- Workspace or storage failure.
- Privileged task or secret access issue.
- Shared task regression.
- Artifact/provenance generation failure.
- Platform-wide queueing or runner exhaustion.

## Response Rules

- Restore safe release capability before optimizing CI convenience.
- Prefer targeted rollback of shared platform changes.
- Preserve task logs, pod events, and artifact evidence for RCA.
- Communicate clearly whether failure is Tekton control plane, cluster resource pressure, or task logic.

## Principal Review Lens

- Can responders restore safe execution quickly?
- Which emergency action most risks supply-chain trust?
- What proves the platform is healthy again?
- Are runbooks realistic for multi-tenant pipeline outages?
