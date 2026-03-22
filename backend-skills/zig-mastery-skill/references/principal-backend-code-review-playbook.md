# Principal Backend Code Review Playbook for Zig Services

## Review Lens

Review for:

- ownership correctness
- overload and queue posture
- rollout compatibility
- tenant fairness and blast radius
- observability sufficiency under incidents

## Questions

- what saturates first under dependency slowness?
- where can memory or queue growth escape control?
- can old and new versions coexist safely?
- what operator signal is still missing?
