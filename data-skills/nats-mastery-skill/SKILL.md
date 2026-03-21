---
name: nats-principal-engineer
description: |
  Principal/Senior-level NATS playbook for subject design, request/reply, JetStream durability, consumer semantics, multi-tenant messaging, and operating lightweight event platforms at scale.
  Use when: designing evented systems with NATS, reviewing JetStream usage, tuning subjects and consumers, or operating NATS in production.
---

# NATS Mastery (Senior → Principal)

## Operate

- Start from messaging semantics: ephemeral pub/sub, request/reply, durable streams, or control-plane signaling.
- Treat subject design, consumer semantics, retention, and topology as first-class architecture concerns.
- Prefer simple subject hierarchies and explicit delivery semantics.
- Design for debuggability and operator clarity before chasing messaging cleverness.

## Default Standards

- Subject naming should encode bounded, meaningful routing intent.
- JetStream retention and ack behavior must match business semantics.
- Consumers must be idempotent where durability and replay matter.
- Multi-tenant usage needs policy before it needs scale.
- Observability should reveal lag, redelivery, and cluster health clearly.

## References

- Subject design and routing semantics: [references/subject-design-and-routing-semantics.md](references/subject-design-and-routing-semantics.md)
- Core NATS pub-sub and request-reply: [references/core-nats-pub-sub-and-request-reply.md](references/core-nats-pub-sub-and-request-reply.md)
- JetStream durability and retention: [references/jetstream-durability-and-retention.md](references/jetstream-durability-and-retention.md)
- Consumer design, acking, and replay: [references/consumer-design-acking-and-replay.md](references/consumer-design-acking-and-replay.md)
- Multi-tenant governance and security: [references/multi-tenant-governance-and-security.md](references/multi-tenant-governance-and-security.md)
- Cluster topology and operations: [references/cluster-topology-and-operations.md](references/cluster-topology-and-operations.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
