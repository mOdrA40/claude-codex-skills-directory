# Multi-Tenant RabbitMQ

## Rules

- Vhost strategy, quotas, and permissions define tenant safety.
- One tenant should not dominate connection count or queue growth silently.
- Tenant-aware operations must remain debuggable and auditable.
- Shared brokers need explicit platform policy.

## Tenancy Heuristics

### Multi-tenancy is a governance discipline first

Shared RabbitMQ is safe only when vhosts, quotas, permissions, naming, and escalation paths make tenant ownership and blast radius clear enough for operators.

### Fairness should be enforceable, not aspirational

If one tenant can create disproportionate queue depth, connection churn, or resource pressure, the platform is depending on social etiquette instead of architecture.

### Incident handling must remain tenant-aware

Responders should be able to isolate one tenant's issue without casually disrupting or exposing the message flows of others.

## Common Failure Modes

### Shared-broker politeness model

The cluster assumes tenants will behave sensibly without enough quotas, lifecycle controls, or clear escalation policy.

### Vhost structure without real governance

Tenancy is represented structurally, but permissions, naming discipline, and operational procedures do not enforce the intended boundary strongly enough.

### Tenant blast radius hidden in aggregates

The broker looks healthy overall until one tenant's queue growth or connection behavior creates the incident that reveals how uneven the platform really is.

## Principal Review Lens

- Which tenant can create the biggest blast radius today?
- How are quotas enforced and monitored?
- Can support isolate one tenant fast during incident response?
- Which multi-tenant shortcut is most likely to become an availability or governance incident later?
