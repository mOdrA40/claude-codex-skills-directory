# Incident Runbooks (Jaeger)

## Cover at Minimum

- Ingest loss or backlog.
- Query degradation.
- Storage backend trouble.
- Bad sampling rollout.
- Tenant-driven overload.
- Broken dashboard/log correlation into traces.

## Response Rules

- Restore the most valuable trace workflows first.
- Prefer targeted mitigation over broad risky backend changes.
- Preserve evidence around dropped traces, sampling changes, and storage errors.
- Communicate clearly when traces are delayed, partial, or unreliable.

## Principal Review Lens

- Can responders recover trustworthy traces quickly?
- Which emergency action most risks making the platform less truthful?
- What proves the system is healthy again end-to-end?
- Are runbooks realistic for shared observability incidents?
