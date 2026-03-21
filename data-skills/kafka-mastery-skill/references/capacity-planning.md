# Capacity Planning (Kafka)

## Rules

- Capacity planning must include retention, replication, and replay spikes.
- Partition count affects throughput, operations, and recovery time.
- Benchmarks should use real message sizes and key distributions.
- Storage and network are often the real budget constraints.

## Capacity Heuristics

### Capacity is shaped by worst-case flow, not averages

Kafka pain usually appears during burst traffic, retention growth, replay waves, broker loss, or skewed hot partitions—not during steady averages.

### Partition count is both throughput and operations budget

More partitions can help concurrency, but they also increase metadata, recovery time, rebalancing complexity, and operational tax.

### Storage and network define recovery realism

A platform with comfortable steady-state metrics may still recover poorly if retention, replication, or replay traffic overwhelms disk and network under failure conditions.

## Common Failure Modes

### Partition optimism

Teams add partitions to chase scale without respecting the recovery and governance cost they also add.

### Replay under-modeled

Normal traffic fits comfortably, but one replay or consumer catch-up event reveals that true headroom was much lower than believed.

### Retention cost hidden in success

Long-lived retained data looks harmless until it magnifies storage cost, restore time, and broker replacement pain.

## Principal Review Lens

- What fails first under 2x event volume?
- Is partition count helping throughput or only adding ops tax?
- How much headroom exists during broker loss or replay storm?
- Which capacity assumption is most likely to fail during a real recovery or reprocessing event?
