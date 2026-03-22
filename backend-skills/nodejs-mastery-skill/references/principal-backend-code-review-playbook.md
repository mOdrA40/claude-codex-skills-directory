# Principal Backend Code Review Playbook for Node.js Services

## Review Lens

A principal review should evaluate:

- correctness under failure
- compatibility during rollout
- tenant fairness and blast radius
- operational visibility
- cost and complexity tradeoffs

## High-Value Questions

- what fails first under dependency slowness?
- can old and new versions coexist safely?
- where is idempotency guaranteed?
- what traffic is protected under overload?
- what would on-call need that this change does not yet provide?

## Anti-Pattern

Approving code that is locally clean but operationally ambiguous.
