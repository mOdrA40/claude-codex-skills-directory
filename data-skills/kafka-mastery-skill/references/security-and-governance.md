# Security and Governance (Kafka)

## Rules

- Topics, ACLs, credentials, and schemas need ownership.
- Multi-team Kafka platforms require governance to avoid chaos.
- PII and regulated data need explicit retention and access controls.
- Protect admin operations and automation paths carefully.

## Governance Heuristics

### Kafka governance is mostly about preventing silent sprawl

Without explicit ownership, topic count, ACL complexity, schema drift, and retention risk all grow faster than teams notice.

### Access design should remain explainable

Operators should be able to answer who can publish, who can consume, who can administer, and which automation paths are effectively privileged.

### Sensitive-event posture must cover lifecycle

Protecting regulated or sensitive data means thinking about publish paths, retention windows, replay behavior, schema fields, and auditability together.

## Common Failure Modes

### ACL growth without clarity

Permissions accumulate over time until nobody can confidently explain the real blast radius of one credential or service account.

### Topic sprawl as governance debt

New topics are easy to create, but ownership, retirement, and consumer visibility stay weak enough that the platform becomes hard to govern.

### Sensitive data normalized in event flows

Teams treat event streams like ordinary transport even when retention, replay, and broad fan-out make sensitive payloads much riskier than in request/response systems.

## Principal Review Lens

- Which team can publish or consume too much today?
- Is topic sprawl becoming an operational risk?
- How are sensitive events audited and retired?
- Which governance shortcut today is most likely to create a security or compliance incident later?
