# Principal Backend Code Review Playbook on the BEAM

## Review Lens

Review for:

- supervision and blast radius correctness
- mailbox and backlog posture
- rollout compatibility
- tenant fairness and dependency failure behavior
- observability and on-call usability

## Questions

- which process family fails first under dependency slowness?
- where can backlog or restart amplification escape control?
- can old and new nodes coexist safely?
- what signal is still missing for triage?
