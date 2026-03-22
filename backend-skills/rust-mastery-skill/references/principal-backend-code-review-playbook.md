# Principal Backend Code Review Playbook for Rust Services

## Review Lens

Review for:

- failure semantics
- cancellation and shutdown correctness
- rollout compatibility
- tenant fairness and blast radius
- observability and operability cost

## Questions

- what fails first under dependency slowness?
- where can async work escape bounds?
- can old and new versions coexist safely?
- what does on-call still not know after this change?
