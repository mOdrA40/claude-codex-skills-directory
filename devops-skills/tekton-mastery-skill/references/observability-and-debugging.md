# Observability and Debugging (Tekton)

## Rules

- Pipeline observability should reveal execution phase, task failure mode, secret/access issues, and artifact handoff state.
- Distinguish task logic failure from platform execution or Kubernetes resource problems quickly.
- Logs and status views should support real incident response, not only happy-path troubleshooting.
- Build platform dashboards should serve both operators and delivery teams.

## Useful Signals

- Task start/finish status, queueing, pod scheduling, retries, workspace issues, artifact generation, and secret mount failures.
- Correlate with cluster events, registry behavior, and deploy systems.
- Standardize debugging workflows for common pipeline failures.
- Preserve execution metadata needed for RCA.

## Principal Review Lens

- Can responders find the failing layer quickly?
- Which missing signal most slows Tekton debugging today?
- Are teams blaming Tekton for task or cluster mistakes too often?
- What observability improvement most reduces MTTR?
