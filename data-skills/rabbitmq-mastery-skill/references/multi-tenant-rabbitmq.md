# Multi-Tenant RabbitMQ

## Rules

- Vhost strategy, quotas, and permissions define tenant safety.
- One tenant should not dominate connection count or queue growth silently.
- Tenant-aware operations must remain debuggable and auditable.
- Shared brokers need explicit platform policy.

## Principal Review Lens

- Which tenant can create the biggest blast radius today?
- How are quotas enforced and monitored?
- Can support isolate one tenant fast during incident response?
