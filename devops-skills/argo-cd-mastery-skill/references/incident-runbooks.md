# Incident Runbooks (Argo CD)

## Cover at Minimum

- Bad sync or auto-sync incident.
- Repo/render access failure.
- Broad drift or ignore-rule mistake.
- Project permission or destination scope issue.
- Shared template/app-of-apps regression.
- Controller degradation.

## Response Rules

- Stabilize critical workloads before restoring full GitOps elegance.
- Prefer targeted pause and rollback over broad emergency mutation.
- Preserve sync history, repo state, and cluster evidence.
- Communicate clearly whether failure is Git, render, Argo, or runtime cluster state.

## Principal Review Lens

- Can responders stop blast radius quickly?
- Which emergency action most risks bigger drift later?
- What proves the control plane is healthy again?
- Are runbooks practical for multi-team delivery incidents?
