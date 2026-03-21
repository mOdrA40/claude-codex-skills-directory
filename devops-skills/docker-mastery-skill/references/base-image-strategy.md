# Base Image Strategy

## Rules

- Base image choice defines patching burden, security posture, and debugability.
- Smaller is not automatically better if operability suffers.
- Pin intentionally and review update cadence.
- Standardize where it reduces risk and cognitive load.

## Principal Review Lens

- Are we choosing a base for fashion or operational fit?
- How painful is emergency patch rollout across all images?
- What runtime assumption comes from this base image?
