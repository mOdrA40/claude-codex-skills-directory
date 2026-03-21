# CMS Integration and Publishing Workflows

## Principle

CMS integration is not only a data problem. It is a publishing, preview, schema, and operational visibility problem.

## Common Failure Modes

- preview flows that diverge too far from production reality
- schema changes breaking content rendering unexpectedly
- editorial workflows with weak validation or ownership

## Workflow Heuristics

### Preview should match publishing reality closely

If editors preview one thing and production renders another, trust in the platform breaks quickly.

### Content schema changes need explicit ownership

Content modeling is a platform concern, not only an editorial convenience. Breaking changes should be reviewed like API changes.

### Publishing incidents need operator visibility

When content fails to render, the team should know whether the fault is in schema, source data, build pipeline, or route logic.

## Review Questions

- what preview behavior is promised to editors?
- how are breaking content model changes detected?
- who owns publishing incidents?
- what content workflow currently hides the most operational risk?
