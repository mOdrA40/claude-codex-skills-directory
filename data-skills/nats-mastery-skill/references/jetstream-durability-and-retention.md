# JetStream Durability and Retention

## Rules

- Stream retention should reflect business semantics, replay needs, and cost.
- Limits, workqueue, and interest retention modes solve different problems.
- Storage choice and replication level influence both reliability and operational tax.
- Retention policy must be visible to consumer owners.

## Design Guidance

- Match stream boundaries to domain and replay behavior.
- Review duplicate publish and idempotency expectations.
- Understand what a retained message means to downstream teams.
- Keep retention growth and storage pressure visible.

## Principal Review Lens

- What user or system promise does this retention mode imply?
- Are we retaining too much because no one owns cleanup?
- Which stream would be hardest to recover correctly?
- Does durability posture match business criticality?
