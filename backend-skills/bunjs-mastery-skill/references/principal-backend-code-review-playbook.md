# Principal Backend Code Review Playbook for Bun Services

## Review Lens

Review beyond code style:

- rollout safety
- dependency failure posture
- observability sufficiency
- tenant fairness
- cost of added complexity

## Questions

- what fails first under dependency slowness?
- does this create backlog or lock amplification risk?
- can old and new versions coexist?
- what would on-call wish this change already exposed?
