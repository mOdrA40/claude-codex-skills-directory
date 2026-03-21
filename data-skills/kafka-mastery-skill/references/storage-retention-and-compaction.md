# Storage, Retention, and Compaction (Kafka)

## Rules

- Retention and compaction policies are part of the product contract.
- Long retention increases storage and recovery burden.
- Compaction semantics must be understood by consumers.
- Tombstones and delete behavior should not surprise downstream teams.

## Retention Heuristics

### Retention is a business promise

How long Kafka keeps data determines what can be replayed, audited, recovered, or reprocessed. That is a product and platform contract, not just a storage knob.

### Compaction changes what history means

Compacted topics support powerful patterns, but they also redefine what downstream teams may assume about prior states, deletions, and replay completeness.

### Storage posture should include recovery realism

Long retention may look safe until it increases broker recovery time, operational cost, and cluster fragility during incident or scale events.

## Common Failure Modes

### Retention by inertia

Topics keep data far longer than necessary because shortening retention feels risky, while the real storage and recovery tax grows silently.

### Compaction misunderstood as full history

Consumers act as if compacted topics preserve everything needed for business replay even when older states and deleted records no longer exist as assumed.

### Tombstone semantics unclear

Delete behavior is technically implemented, but downstream consumers do not share a consistent interpretation of what a tombstone means operationally.

## Principal Review Lens

- Why is this retention period justified?
- What breaks if compaction removes expected history?
- Is storage growth predictable under worst-case traffic?
- Which retention or compaction assumption is most likely to fail during a real recovery event?
