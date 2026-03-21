---
name: kafka-principal-engineer
description: |
  Principal/Senior-level Kafka playbook for topic design, consumer groups, delivery semantics, partitioning, retention, observability, and operating event-driven systems at scale.
  Use when: designing event streams, reviewing partition strategy, debugging lag/rebalance issues, or hardening Kafka in production.
---

# Kafka Mastery (Senior → Principal)

## Operate

- Confirm whether Kafka is solving event streaming, integration decoupling, replayability, or auditability.
- Design topics, keys, and retention with business semantics and recovery strategy in mind.
- Treat consumer lag, rebalance behavior, and schema evolution as first-class concerns.
- Prefer explicit delivery semantics over magical “exactly once” assumptions.

## Default Standards

- Partition by stable business keys with clear ordering needs.
- Consumers must be idempotent.
- Retention and compaction are product decisions, not only infra settings.
- Observe lag, throughput, error rate, rebalance churn, and ISR health.

## References

- Topic design and delivery semantics: [references/topic-design-and-delivery.md](references/topic-design-and-delivery.md)
- Consumers and operations: [references/consumers-and-operations.md](references/consumers-and-operations.md)
- Partitioning and ordering: [references/partitioning-and-ordering.md](references/partitioning-and-ordering.md)
- Schema evolution: [references/schema-evolution.md](references/schema-evolution.md)
- Producers: [references/producers.md](references/producers.md)
- Consumer groups: [references/consumer-groups.md](references/consumer-groups.md)
- Delivery semantics: [references/delivery-semantics.md](references/delivery-semantics.md)
- Retries, DLQ, and replay: [references/retries-dlq-and-replay.md](references/retries-dlq-and-replay.md)
- Storage, retention, and compaction: [references/storage-retention-and-compaction.md](references/storage-retention-and-compaction.md)
- Broker and cluster operations: [references/broker-and-cluster-operations.md](references/broker-and-cluster-operations.md)
- Observability: [references/observability.md](references/observability.md)
- Security and governance: [references/security-and-governance.md](references/security-and-governance.md)
- Capacity planning: [references/capacity-planning.md](references/capacity-planning.md)
- Multi-tenant Kafka: [references/multi-tenant-kafka.md](references/multi-tenant-kafka.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
