# Capacity Planning

## Rules

- Plan for queue depth, burst publish rate, consumer throughput, and failure headroom.
- Memory, disk alarms, and quorum overhead all matter.
- Benchmarks should model real message size and acknowledgement behavior.
- HA topology changes capacity math materially.

## Capacity Heuristics

### Capacity is mostly about worst-case shape

RabbitMQ rarely fails because average load was misunderstood. It fails because burst size, backlog growth, message shape, ack timing, or HA overhead were weaker than assumed.

### Headroom should include recovery conditions

The platform must survive not only steady traffic, but also node loss, consumer slowdown, replay bursts, and backlog drain periods without collapsing into alarm states.

### Message behavior matters as much as count

Message size, persistence, routing complexity, ack strategy, and queue type can change capacity posture more than raw message-per-second comparisons suggest.

## Common Failure Modes

### Average-load comfort

The cluster looks well-sized until one realistic burst, one slow consumer class, or one node event reveals much weaker true headroom.

### Quorum overhead under-modeled

Durability expectations rise, but the cost of quorum behavior on throughput and recovery was not internalized.

### Backlog drain fantasy

The system can accumulate backlog faster than it can safely clear it once consumers or dependencies recover.

## Principal Review Lens

- What fails first under burst traffic?
- How much headroom remains during node loss?
- Which queue or tenant dominates platform cost?
- Which capacity assumption is most likely to break during a replay or recovery event?
