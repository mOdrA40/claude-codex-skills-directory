# Change Safety, Review, and Rollout Workflows

## Rules

- Automation changes should be reviewable in terms of operational effect, not only syntax.
- Dry-run, diff, scoping, and staged execution are key safety mechanisms.
- High-blast-radius playbooks require stronger review and narrower rollout.
- Rollback or remediation posture should exist before risky changes execute.

## Good Practices

- Highlight tasks that restart services, alter security posture, or mutate data.
- Scope execution by inventory slice, tags, or canary hosts when possible.
- Keep change windows aligned with risk and operator availability.
- Preserve execution logs and context for RCA.

## Principal Review Lens

- What is the first host or group this should touch, and why?
- Which task could succeed syntactically but still break production?
- Can reviewers understand operational impact before execution?
- What one change would most reduce rollout risk?
