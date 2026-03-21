# Debugging Containers

## Rules

- Debugging should start from image, runtime config, and environment assumptions.
- Distinguish app issues from containerization issues quickly.
- Prefer immutable debugging patterns over ad-hoc shell surgery.
- Logs, inspect output, and resource signals should be first-class tools.

## Principal Review Lens

- Is this bug caused by the app, image, or runtime environment?
- What signal most quickly narrows the problem?
- Are we debugging in a way that can be repeated safely?
