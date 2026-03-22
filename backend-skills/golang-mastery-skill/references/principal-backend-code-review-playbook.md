# Principal Backend Code Review Playbook for Go Services

## Review Lens

Review for:

- failure-mode correctness
- concurrency and shutdown safety
- rollout compatibility
- observability and on-call usability
- tenant fairness and cost impact

## Questions

- what fails first under dependency slowness?
- where can goroutine or queue growth escape bounds?
- can old and new binaries coexist safely?
- what signal is missing for incident response?
